from skimage import io, transform
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

def show_mosaic_images(root_dir: str) -> None:
    """
    Displays a mosaic of images from the specified root directory.
    
    Parameters:
    - root_dir (str): Path to the root directory containing class folders.
    """
    rows, cols = 5, 4
    image_shape = (256, 256)
    images = []

    for class_folder in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_folder)
        
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path)[:2]:
                image_path = os.path.join(class_path, image_name)
                img = io.imread(image_path)
                
                img_resized = transform.resize(img, image_shape, anti_aliasing=True)
                images.append(img_resized)
                
                if len(images) == rows * cols:
                    break
            if len(images) == rows * cols:
                break
        if len(images) == rows * cols:
            break

    # Criar o mosaico de imagens
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    for ax, img in zip(axes.flatten(), images):
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def generate_mosaico_from_object(class_codification_tabel:list, dataset, category_img,
                                 cols = 5, rows = 2 ) -> None:

    plt.figure(figsize=(15, 6))

    for i, (category, _) in enumerate(class_codification_tabel):
        data_imgs = dataset.get_category_images(category)  
        
        if len(data_imgs) > 0:  
            img_data = data_imgs[0]  
            type = img_data[category_img]  
            
            mask_rgb = cv2.cvtColor(type, cv2.COLOR_GRAY2RGB)
            

            plt.subplot(rows, cols, i + 1)
            plt.imshow(mask_rgb)
            plt.title(category)  
            plt.axis('off')  

    plt.tight_layout()
    plt.show()