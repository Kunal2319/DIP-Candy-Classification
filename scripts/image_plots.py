import os
import cv2
import re
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_mosaic_from_dir(dir_path: str, images_per_class=2) -> None:
    """
    Displays a mosaic of images from the specified root directory, with 2 images per class.

    Args:
        dir_path (str): Path to the root directory containing the images.
        images_per_class (int): Number of images to display per class.
    """
    rows, cols = 5, images_per_class * 2
    image_shape = (256, 256)
    images = []
    class_images = {}

    # Iterate through images in the directory and group them by class
    for image_name in os.listdir(dir_path):
        # Extract class code from image name using regex
        match = re.match(r'^\d+', image_name)
        if match is None:
            continue
        class_code = match.group()

        # Initialize list for each class if not already
        if class_code not in class_images:
            class_images[class_code] = []

        if len(class_images[class_code]) < images_per_class:
            image_path = os.path.join(dir_path, image_name)
            image = io.imread(image_path)
            image = transform.resize(image, image_shape)
            class_images[class_code].append(image)
        
        if len(class_images) == rows * cols:
            break

    # Flatten the list of images in class order
    for class_code in sorted(class_images.keys()):
        images.extend(class_images[class_code])

    # Plot the images in a grid
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    for ax, img in zip(axes.flatten(), images):
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_mosaic_from_dataset(dataset, data_type='images', plot_bbox=False, cols=5, rows=2) -> None:
    """
    Plot a mosaic of images or masks from the dataset. The images and masks are randomly selected, one per class.

    Args:
        dataset (CandyDataset): The dataset object.
        data_type (str): The type of data to plot. It can be 'images' or 'masks'.
        plot_bbox (bool): Whether to plot the bounding box or not.
        cols (int): The number of columns in the plot.
        rows (int): The number of rows in the plot.
    """
    plt.figure(figsize=(15, 6))
    classes = dataset.categories_indexes.keys()
    for i, class_name in enumerate(classes):

        class_indexes = dataset.categories_indexes[class_name]
        images_data = dataset.get_images_data(class_indexes)
        random_image = np.random.choice(images_data)
    
        image = random_image['image']
        mask = random_image['mask'] * 255

        plt.subplot(rows, cols, i + 1)
        plt.imshow(image if data_type == 'images' else mask, cmap='gray')

        if plot_bbox:
            bbox = random_image['bbox']
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                     linewidth=2, edgecolor='g', facecolor='none')
            plt.gca().add_patch(rect)

        plt.title(class_name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
