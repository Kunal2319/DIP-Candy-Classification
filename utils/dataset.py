import os
import cv2
import json
import uuid
import numpy as np
import collections
import albumentations as A


class CandyDataset:
    def __init__(self, data_dir) -> None:
        """
        Initialize the dataset with the directory containing the images and annotations.

        Args:
            data_dir (str): Directory containing image files and JSON annotation files.
        """
        self.data_dir = data_dir
        self.images = []
        self.masks = []
        self.bboxes = []

        self._json = None
        self._last_id = 0
        self._categories = {}

        # Dictionary to store indexes of images for each split
        self.data_split = {
            'train': [],
            'val': [],
            'test': []
        }

        # Dictionary to store indexes of images for each category
        # Key: category_id, Value: List of image indexes
        self.categories_indexes = collections.defaultdict(list)

        self.load_data()

    def load_data(self):
        """
        Load JSON annotations and images from the specified directory.
        """
        for file in os.listdir(self.data_dir):
            if file.endswith('.json'):
                with open(os.path.join(self.data_dir, file)) as f:
                    self._json = json.load(f)

                    # Fill the categories dictionary
                    for category in self._json['categories']:
                        self._categories[category['id']] = category['name']

                    self._load_annotations(self._json)

    def _load_annotations(self, data) -> None:
        """
        Load image annotations from a JSON file and prepare corresponding images, masks, and bounding boxes.

        Args:
            data (dict): Parsed JSON data with annotations and image information.
        """
        for img_data in data['images']:
            img_path = os.path.join(self.data_dir, img_data['file_name'])
            img = cv2.imread(img_path)

            if img_path.endswith('.json') or img is None:
                print(f'Error reading image: {img_path}')
                continue

            ann = data['annotations'][img_data['id'] - 1]

            category_name = self._categories[ann['category_id']]
            self.categories_indexes[category_name].append(len(self.images))

            rle_mask = ann['segmentation']['counts']
            mask = self._rle_to_mask(
                rle_mask, (img_data['height'], img_data['width']))
            bbox = np.array([ann['bbox']])

            self.images.append(img)
            self.masks.append(mask)
            self.bboxes.append(bbox)
            self._last_id = max(self._last_id, img_data['id'])

    def add_sample(self, img, img_name: str, bbox: list, mask, img_id: int, category_id: int) -> None:
        """
        Add a new image sample with bounding box and mask to the dataset.

        Args:
            img (numpy.ndarray): Image data.
            img_name (str): Image filename.
            bbox (list): Bounding box coordinates.
            mask (numpy.ndarray): Segmentation mask.
            img_id (int): Image identifier.
            category_id (int): Category identifier.
        """
        segmentation = self._mask_to_rle(mask)
        self._last_id += 1

        self._json['annotations'].append({
            'id': self._last_id,
            'image_id': img_id,
            'category_id': category_id,
            'area': bbox[2] * bbox[3],
            'bbox': bbox,
            'iscrowd': 0,
            'attributes': {
                'occluded': False,
                'rotation': 0.0
            },
            'segmentation': {
                'counts': segmentation,
                'size': [img.shape[0], img.shape[1]]
            }
        })

        self._json['images'].append({
            'id': img_id,
            'file_name': img_name,
            'width': img.shape[1],
            'height': img.shape[0],
            'license': 0,
            'flickr_url': '',
            'coco_url': '',
            'date_captured': 0
        })

        category_name = self._categories[category_id]
        self.categories_indexes[category_name].append(len(self.images))

        self.images.append(img)
        self.masks.append(mask)
        self.bboxes.append(bbox)

    def apply_augmentation(self, image, bboxes: list, mask, angle=15, shear_value=15):
        """
        Apply data augmentation to the given image, bounding boxes, and mask.

        Args:
            image (numpy.ndarray): Image to augment.
            bboxes (list): List of bounding boxes.
            mask (numpy.ndarray): Segmentation mask.
            angle (int, optional): Rotation angle limit. Defaults to 15.
            shear_value (int, optional): Shear transformation limit. Defaults to 15.

        Returns:
            dict: Augmented image, bounding boxes, and mask.
        """
        transform = A.Compose([
            A.OneOf([
                A.SafeRotate(limit=angle, border_mode=cv2.BORDER_REPLICATE),
                A.Affine(shear=shear_value, mode=cv2.BORDER_REPLICATE),
                A.RandomRotate90(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.NoOp(),
            ]),
            A.RandomBrightnessContrast(
                p=0.5, brightness_limit=0.1, contrast_limit=0.1),
            A.GaussianBlur(p=0.5),
            A.HueSaturationValue(p=0.5, hue_shift_limit=0,
                                 sat_shift_limit=10, val_shift_limit=0)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

        result = transform(image=image, bboxes=bboxes, mask=mask, labels=[0])
        result['bbox'] = result['bboxes'][0]
        return result

    def augment_dataset(self, num_copies=8) -> None:
        """
        Apply augmentation to the entire dataset to generate multiple copies of each image.

        Args:
            num_copies (int, optional): Number of augmented copies to generate for each image. Defaults to 8.
        """
        json_copy = self._json.copy()
        images_copy = self.images.copy()

        for i, image in enumerate(images_copy):
            mask = self.masks[i]
            bbox = np.array(self.bboxes[i])
            img_id = json_copy['images'][i]['id']
            category_id = json_copy['annotations'][i]['category_id']

            img_name = json_copy['images'][i]['file_name'].split('.')[0]
            for _ in range(num_copies):
                augmented_data = self.apply_augmentation(image, bbox, mask)
                code = uuid.uuid4().hex[:4]
                augmented_name = f'{img_name}-augmented-{code}.jpg'

                self.add_sample(
                    augmented_data['image'],
                    augmented_name,
                    augmented_data['bbox'],
                    augmented_data['mask'],
                    img_id,
                    category_id
                )

    def equalize_histogram(self, image, clip_limit=1.0, tile_grid_size=(8, 8)):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance image contrast.

        Args:
            image (numpy.ndarray): Input image.
            clip_limit (float, optional): Threshold for contrast clipping. Defaults to 1.0.
            tile_grid_size (tuple, optional): Grid size for histogram equalization. Defaults to (8, 8).

        Returns:
            numpy.ndarray: Image with equalized contrast.
        """
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=tile_grid_size)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    def normalize_dataset(self) -> None:
        """
        Apply histogram equalization to the entire dataset to normalize image contrast.
        """
        for i, image in enumerate(self.images):
            self.images[i] = self.equalize_histogram(image)

    def split_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Split the dataset into training, validation, and test sets.

        Args:
            train_ratio (float, optional): Training set ratio. Defaults to 0.8.
            val_ratio (float, optional): Validation set ratio. Defaults to 0.1.
            test_ratio (float, optional): Test set ratio. Defaults to 0.1.
        """

        num_images = len(self.images)
        num_train = int(train_ratio * num_images)
        num_val = int(val_ratio * num_images)

        indices = np.random.permutation(num_images)
        self.data_split['train'] = indices[:num_train]
        self.data_split['val'] = indices[num_train:num_train + num_val]
        self.data_split['test'] = indices[num_train + num_val:]

    def get_images_data(self, indexes: list) -> list:
        """
        Get images from the dataset by index.

        Args:
            indexes (list): List of indexes to retrieve.

        Returns:
            list: List of images.
        """
        images_data = []
        for idx in indexes:
            images_data.append(self.__getitem__(idx))
        return images_data

    def get_category_images(self, category_name: str) -> list:
        """
        Get image indexes for a specific category.

        Args:
            category_name (str): Category name.

        Returns:
            list: List of images.
        """
        indexes = self.categories_indexes[category_name]
        images_data = self.get_images_data(indexes)      

        return images_data

    def export_images(self, output_dir: str) -> None:
        """
        Export the dataset images to the specified output directory.

        Args:
            output_dir (str): Directory to save images.
        """
        os.makedirs(output_dir, exist_ok=True)
        for i in range(len(self.images)):
            img_name = self._json['images'][i]['file_name']
            img_path = os.path.join(output_dir, img_name)
            img = self.images[i]
            cv2.imwrite(img_path, img)

    def export_masks(self, output_dir: str) -> None:
        """
        Export the masks to the specified output directory.

        Args:
            output_dir (str): Directory to save masks.
        """
        os.makedirs(output_dir, exist_ok=True)
        for i in range(len(self.images)):
            img_name = self._json['images'][i]['file_name'].split('.')[0]
            mask_name = f'{img_name}-mask.jpg'
            mask = self.masks[i] * 255
            mask_path = os.path.join(output_dir, mask_name)
            cv2.imwrite(mask_path, mask)

    def save_annotations(self, output_dir: str) -> None:
        """
        Save the JSON annotations to the specified output directory.

        Args:
            output_dir (str): Directory to save the annotations JSON file.
        """
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, '_annotations.json'), 'w') as f:
            json.dump(self._json, f, indent=4)

    def _rle_to_mask(self, rle_counts: list, mask_size: tuple):
        """
        Convert RLE (Run Length Encoding) to a binary mask.

        Args:
            rle_counts (list): RLE encoded segmentation.
            mask_size (tuple): Size of the mask.

        Returns:
            numpy.ndarray: Binary mask.
        """
        height, width = mask_size
        mask = np.zeros(height * width, dtype=np.uint8)

        current_pos = 0
        for i, count in enumerate(rle_counts):
            count = int(count)
            if i % 2 == 0:
                current_pos += count
            else:
                mask[current_pos:current_pos + count] = 1
                current_pos += count
        mask = mask.reshape((width, height)).T
        return mask.reshape((height, width))

    def _mask_to_rle(self, mask) -> list:
        """
        Convert a binary mask to RLE (Run Length Encoding).

        Args:
            mask (numpy.ndarray): Binary mask.

        Returns:
            list: RLE encoded mask.
        """
        pixels = mask.flatten(order='F')
        prev_pixel, count = 0, 0
        rle = []

        for pixel in pixels:
            if pixel == prev_pixel:
                count += 1
            else:
                rle.append(count)
                count = 1
            prev_pixel = pixel
        rle.append(count)

        if pixels[0] == 1:
            rle = [0] + rle

        return rle

    def __getitem__(self, idx: int):
        """
        Get an item from the dataset by index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: Image, mask, and bounding box data.
        """
        category_id = self._json['annotations'][idx]['category_id']
        return {
            'image': self.images[idx],
            'mask': self.masks[idx],
            'bbox': self.bboxes[idx],
            'category': self._categories[category_id],
            'category_id': category_id
        }