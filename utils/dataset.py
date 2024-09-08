import os
import cv2
import numpy as np
import albumentations as A
import json
import uuid


class CandyDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images = []
        self.masks = []
        self.bboxes = []

        self._json = None
        self._last_id = 0

        self.load_data()

    def load_data(self):
        for file in os.listdir(self.data_dir):
            if file.endswith('.json'):
                with open(os.path.join(self.data_dir, file)) as f:
                    self._json = json.load(f)
                    self._load_annotations(self._json)

    def _load_annotations(self, data):
        for img_data in data['images']:
            img_path = os.path.join(self.data_dir, img_data['file_name'])
            img = cv2.imread(img_path)

            if img_path.endswith('.json'):
                continue

            if img is None:
                print(f'Error reading image: {img_path}')
                continue

            ann = data['annotations'][img_data['id'] - 1]
 
            rle_mask = ann['segmentation']['counts']
            mask = self._rle_to_mask(rle_mask, (img_data['height'], img_data['width']))
            bbox = np.array([ann['bbox']])

            self.images.append(img)
            self.masks.append(mask)
            self.bboxes.append(bbox)
            self._last_id = max(self._last_id, img_data['id'])

    def add_sample(self, img, img_name, bbox, mask, img_id, category_id):
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

        self.images.append(img)
        self.masks.append(mask)
        self.bboxes.append(bbox)
    
    def apply_augmentation(self, image, bboxes, mask, angle=15, shear_value=15):
        transform = A.Compose([
            A.OneOf([
                A.SafeRotate(limit=angle, border_mode=cv2.BORDER_REPLICATE),
                A.Affine(shear=shear_value, mode=cv2.BORDER_REPLICATE),
                A.RandomRotate90(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.NoOp(),
            ]),
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.1, contrast_limit=0.1),
            A.GaussianBlur(p=0.5),
            A.HueSaturationValue(p=0.5, hue_shift_limit=0, sat_shift_limit=10, val_shift_limit=0)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

        result = transform(image=image, bboxes=bboxes, mask=mask, labels=[0])
        result['bbox'] = result['bboxes'][0]
        return result

    def augment_dataset(self, num_copies=8):
        json_copy = self._json.copy()
        images_copy = self.images.copy()

        for i, image in enumerate(images_copy):
            mask = self.masks[i]
            bbox = np.array(self.bboxes[i])
            img_id = json_copy['images'][i]['id']

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
                    augmented_data['labels'][0]
                )
    
    def equalize_histogram(self, image, clip_limit=1.0, tile_grid_size=(8, 8)):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    def normalize_dataset(self):
        for i, image in enumerate(self.images):
            self.images[i] = self.equalize_histogram(image)

    def export_images(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for i in range(len(self.images)):
            img_name = self._json['images'][i]['file_name']
            img_path = os.path.join(output_dir, img_name)
            img = self.images[i]
            cv2.imwrite(img_path, img)

    def export_masks(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for i in range(len(self.images)):
            img_name = self._json['images'][i]['file_name'].split('.')[0]
            mask_name = f'{img_name}-mask.jpg'
            mask = self.masks[i] * 255
            mask_path = os.path.join(output_dir, mask_name)
            cv2.imwrite(mask_path, mask)

    def save_annotations(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, '_annotations.json'), 'w') as f:
            json.dump(self._json, f, indent=4)

    def _rle_to_mask(self, rle_counts, mask_size):
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

    def _mask_to_rle(self, mask):
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
