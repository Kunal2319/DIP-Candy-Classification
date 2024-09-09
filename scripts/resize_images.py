import os
from PIL import Image

def resize_images( input_dir: str, output_dir: str, target_size) -> None:
    """
    Resizes images in the specified input directory and saves them to the output directory.
    
    Args:
        - input_dir (str): Path to the directory containing the images to be resized.
        - output_dir (str): Path to the directory where resized images will be saved.
        - target_size (tuple): Desired size for resizing images, e.g., (256, 256).
    """
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')): #open files
                input_path = os.path.join(root, file)
                
                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, relative_path, file) #save in a separated folder
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                with Image.open(input_path) as img:
                    img = img.resize(target_size, Image.Resampling.LANCZOS) # resized 
                    
                    if img.mode == 'RGBA': #special case 
                        img = img.convert('RGB')
                    
                    img.save(output_path)
