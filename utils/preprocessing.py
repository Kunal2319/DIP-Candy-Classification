import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


'''
Some images with black background from the dataset have inconsistencies in the background, having variations in texture and brightness.
This script corrects these inconsistencies, darkening and blurring the background to make it more uniform while keeping the object in the foreground.
'''


def correct_image_background(
        image, threshold=50,
        kernel_ed_size=(3, 3), kernel_oc_size=(3, 3), iterations=(3, 3),
        blur_size=(9, 9), darken_weight=0.5):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_not(mask)

    kernel_ed = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_ed_size)
    mask = cv2.erode(mask, kernel_ed, iterations=iterations[0])

    if kernel_oc_size != (0, 0):
        kernel_oc = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_oc_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_oc)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_oc)
    mask = cv2.dilate(mask, kernel_ed, iterations=iterations[1])

    blurred_background = cv2.GaussianBlur(image, blur_size, 0)

    dark_background = blurred_background * darken_weight
    final_image = np.where(mask[..., None] == 255, image, dark_background)
    final_image = final_image.astype(np.uint8)

    return final_image


def correct_background(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        black_images = filename.endswith('B.jpg')
        if img is None:
            continue

        is_banana = filename.startswith('00')
        is_bear = filename.startswith('01')
        is_blackberry = filename.startswith('02')
        is_brick = filename.startswith('03')
        is_car = filename.startswith('04')
        is_mouth = filename.startswith('05')
        is_plane = filename.startswith('06')
        is_strawberry = filename.startswith('07')
        is_watermellon = filename.startswith('08')
        is_worm = filename.startswith('09')

        if black_images:
            if is_banana:
                final_image = correct_image_background(img, threshold=60)
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, final_image)

            elif is_bear:
                final_image = correct_image_background(img, darken_weight=0.3)

            elif is_blackberry:
                final_image = correct_image_background(img,
                                                       threshold=46,
                                                       kernel_oc_size=(4, 4),
                                                       iterations=(2, 5),
                                                       blur_size=(13, 13),
                                                       )

            elif is_brick:
                final_image = correct_image_background(img,
                                                       threshold=61,
                                                       blur_size=(7, 7),
                                                       darken_weight=0.4,
                                                       )

            elif is_car:
                final_image = correct_image_background(img)

            elif is_mouth:
                final_image = correct_image_background(img, threshold=65)

            elif is_plane:
                final_image = correct_image_background(img,
                                                       threshold=40,
                                                       kernel_oc_size=(4, 4),
                                                       iterations=(1, 3),
                                                       blur_size=(7, 7),
                                                       darken_weight=0.6,
                                                       )

            elif is_strawberry:
                final_image = correct_image_background(img,
                                                       iterations=(2, 4),
                                                       darken_weight=0.6,
                                                       )

            elif is_watermellon:
                final_image = correct_image_background(img,
                                                       threshold=35,
                                                       iterations=(2, 4),
                                                       darken_weight=0.6,
                                                       )

            elif is_worm:
                final_image = correct_image_background(
                    img, threshold=60, darken_weight=0.4
                )
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, final_image)
        else:
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.getcwd())
    dataset_dir = os.path.join(base_dir, 'dataset')

    data_path = os.path.join(dataset_dir, '0 - original')
    output_dir = os.path.join(dataset_dir, '1 - preprocessed')

    correct_background(data_path, output_dir)
