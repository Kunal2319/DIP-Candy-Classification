import itertools
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
        """
        self.data_dir = data_dir
        self.images, self.masks, self.bboxes = [], [], []
        self._json = None
        self._last_id = 0
        self._categories = {}
        self.data_split = {'train': [], 'val': [], 'test': []}
        self.categories_indexes = collections.defaultdict(list)

        self._load_data()

    def _load_data(self):
        """
        Load the dataset images and annotations from the JSON file.
        """
        # Find the JSON file in the data directory
        json_file = next((f for f in os.listdir(self.data_dir)
                         if f.endswith('.json')), None)
        if not json_file:
            raise FileNotFoundError("No JSON file found.")

        with open(os.path.join(self.data_dir, json_file)) as f:
            self._json = json.load(f)

        self._categories = {
            cat['id']: cat['name'] for cat in self._json['categories']
        }

        for img_data in self._json['images']:
            img_path = os.path.join(self.data_dir, img_data['file_name'])
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error reading image: {img_path}")
                continue

            ann = self._json['annotations'][img_data['id'] - 1]
            category_name = self._categories[ann['category_id']]
            self.categories_indexes[category_name].append(img_data['id'] - 1)

            rle_mask = ann['segmentation']['counts']
            mask = self._rle_to_mask(
                rle_mask, (img_data['height'], img_data['width']))
            bbox = ann['bbox']

            self.images.append(img)
            self.masks.append(mask)
            self.bboxes.append(bbox)
            self._last_id = max(self._last_id, img_data['id'])

    def add_sample(self, img, img_name: str, bbox: list, mask, category_id: int) -> None:
        """
        Add a new image sample with bounding box and mask to the dataset.
        """
        img_id = self._last_id + 1
        self._last_id = img_id

        segmentation = self._mask_to_rle(mask)
        annotation = {
            'id': img_id,
            'image_id': img_id,
            'category_id': category_id,
            'area': bbox[2] * bbox[3],
            'bbox': bbox,
            'iscrowd': 0,
            'segmentation': {'counts': segmentation, 'size': [img.shape[0], img.shape[1]]}
        }
        self._json['annotations'].append(annotation)

        image_info = {
            'id': img_id,
            'file_name': img_name,
            'width': img.shape[1],
            'height': img.shape[0]
        }
        self._json['images'].append(image_info)

        category_name = self._categories[category_id]
        self.categories_indexes[category_name].append(img_id - 1)
        self.images.append(img)
        self.masks.append(mask)
        self.bboxes.append(bbox)

    def apply_augmentation(self, image, bbox: list, mask, angle=15, shear_value=15) -> dict:
        """
        Apply data augmentation to the given image, bounding boxes, and mask. 
        Each augmentation operation is applied with a probability of 0.5.

        The operations are:
        - Random rotation
        - Random shear transformation
        - Random rotation by 90 degrees
        - Horizontal and Vertical flip
        - Random brightness and contrast
        - Gaussian blur
        - Saturation shift

        Args:
            image (numpy.ndarray): Image to augment.
            bboxes (list): List of bounding boxes.
            mask (numpy.ndarray): Segmentation mask.
            angle (int, optional): Rotation angle limit. Defaults to 15.
            shear_value (int, optional): Shear transformation limit. Defaults to 15.

        Returns:
            dict: Augmented image, bounding boxes, and mask.
        """
        bboxes = [bbox]
        transform = A.Compose([
            A.OneOf([
                A.SafeRotate(limit=angle, border_mode=cv2.BORDER_REPLICATE),
                A.Affine(shear=shear_value, mode=cv2.BORDER_REPLICATE),
                A.RandomRotate90(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
            ]),
            A.RandomBrightnessContrast(
                p=0.5, brightness_limit=0.1, contrast_limit=0.1),
            A.GaussianBlur(p=0.5),
            A.HueSaturationValue(p=0.5, hue_shift_limit=0,
                                 sat_shift_limit=10, val_shift_limit=0)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

        result = transform(image=image, bboxes=bboxes, mask=mask, labels=[0])
        result['bbox'] = [float(round(b)) for b in result['bboxes'][0]]
        return result

    def augment_dataset(self, num_copies=8) -> None:
        """
        Apply augmentation to the entire dataset to generate multiple copies of each image.
        """
        for i, image in enumerate(self.images.copy()):
            mask = self.masks[i]
            bbox = self.bboxes[i]
            category_id = self._json['annotations'][i]['category_id']
            img_name = self._json['images'][i]['file_name'].split('.')[0]

            for _ in range(num_copies):
                augmented_data = self.apply_augmentation(image, bbox, mask)
                augmented_name = f'{img_name}-augmented-{uuid.uuid4().hex[:4]}.jpg'
                self.add_sample(augmented_data['image'], augmented_name,
                                augmented_data['bbox'], augmented_data['mask'], category_id)

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
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    def normalize_dataset(self) -> None:
        """
        Apply histogram equalization to the entire dataset to normalize image contrast.
        """
        self.images = [self.equalize_histogram(image) for image in self.images]

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
        return [self.__getitem__(idx) for idx in indexes]

    def export_data(self, output_dir: str, data_type='images') -> None:
        """
        Export the dataset images to the specified output directory.

        Args:
            output_dir (str): Directory to save the images.
            data_type (str, optional): Type of data to export ('images' or 'masks'). Defaults to 'images'.
        """
        os.makedirs(output_dir, exist_ok=True)
        for i, img_name in enumerate(self._json['images']):
            name = img_name['file_name'].split('.')[0]
            file_name = f'{name}-mask.jpg' if data_type == 'masks' else img_name['file_name']

            if data_type == 'masks':
                data = self.masks[i] * 255
            else:
                data = self.images[i]
            cv2.imwrite(os.path.join(output_dir, file_name), data)

    def save_annotations(self, output_dir: str) -> None:
        """
        Save the JSON annotations to the specified output directory.
        The annotations are saved in a file named '_annotations.json'.
        """
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, '_annotations.json'), 'w') as f:
            json.dump(self._json, f, indent=4)

    def _rle_to_mask(self, rle_code: list, mask_shape: tuple):
        """
        Convert RLE (Run Length Encoding) to a binary mask.

        Args:
            rle_counts (list): RLE encoded segmentation.
            mask_size (tuple): Size of the mask.

        Returns:
            numpy.ndarray: Binary mask.
        """
        mask = np.zeros(mask_shape[0] * mask_shape[1], dtype=np.uint8)
        current_pos = 0
        for i, count in enumerate(rle_code):
            if i % 2 == 0:
                current_pos += count
            else:
                mask[current_pos:current_pos + count] = 1
                current_pos += count
        return mask.reshape(mask_shape, order='F')

    def _mask_to_rle(self, mask) -> list:
        """
        Convert a binary mask to RLE (Run Length Encoding).

        Args:
            mask (numpy.ndarray): Binary mask.

        Returns:
            list: RLE encoded mask.
        """
        pixels = mask.flatten(order='F')
        rle = [sum(1 for _ in group)
               for pixel, group in itertools.groupby(pixels)]
        rle = [0] + rle if pixels[0] else rle
        return rle

    def __getitem__(self, idx: int):
        """
        Get an item from the dataset by index.

        Returns:
            dict: Image, mask, bounding box, category, and category ID.
        """
        category_id = self._json['annotations'][idx]['category_id']
        return {
            'image': cv2.cvtColor(self.images[idx], cv2.COLOR_BGR2RGB),
            'mask': self.masks[idx],
            'bbox': self.bboxes[idx],
            'category': self._categories[category_id],
            'category_id': category_id
        }
