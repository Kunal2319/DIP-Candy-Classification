from skimage import io, transform
import os
import matplotlib.pyplot as plt
import numpy as np

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
            for subfolder in ['W', 'B']: #get 1 pic from White and Black
                subfolder_path = os.path.join(class_path, subfolder)
                
                if os.path.isdir(subfolder_path): # open subfolders 
                    for image_name in os.listdir(subfolder_path)[:1]:
                        image_path = os.path.join(subfolder_path, image_name)
                        img = io.imread(image_path)
                        
                        img_resized = transform.resize(img, image_shape, anti_aliasing=True)
                        images.append(img_resized)
                        
                        if len(images) == 20:
                            break
                if len(images) == 20:
                    break
        if len(images) == 20:
            break

    #show image matrix 4 x 5
    if len(images) == rows * cols:
        mosaic = np.vstack([np.hstack(images[i * cols:(i + 1) * cols]) for i in range(rows)])
        plt.figure(figsize=(10, 8))
        plt.imshow(mosaic)
        plt.axis('off')
        plt.show()
    else:
        print(f"Insufficient number of images to create the mosaic. Only {len(images)} images found.")
