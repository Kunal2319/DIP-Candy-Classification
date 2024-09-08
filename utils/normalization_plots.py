import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# prototipo medio 
def average_prototype(red_images: list, green_images: list, blue_images: list): 
    """
    Calculate the average prototype of a class, which is the mean of all images.

    Args:
        red_images: List of NumPy arrays representing the red channel of images.
        green_images: List of NumPy arrays representing the green channel of images.
        blue_images: List of NumPy arrays representing the blue channel of images.

    Returns:
        prototype: NumPy array representing the average prototype image.
    """
    mean_red = np.mean(red_images, axis=0)
    mean_green = np.mean(green_images, axis=0)
    mean_blue = np.mean(blue_images, axis=0)

    # Reconstruct the RGB image by combining the three channels
    average_prototype = np.stack((mean_red, mean_green, mean_blue), axis=-1)

    # Ensure values are within 0-255 range (if normalized between 0 and 1, adjust if necessary)
    average_prototype = np.clip(average_prototype, 0, 255).astype(np.uint8)

    return average_prototype

# histograma_medio_e_variancia
def histogram_mean_and_variance(average_prototype, red_images: list, green_images: list, 
                                blue_images: list, hue_images: list,
                                saturation_images: list, value_images: list, name: str, bins=256):
    """
    Calculate and plot the mean histogram and variance of histograms for a list of images,
    alongside the average prototype image.

    Args:
        average_prototype: NumPy array representing the average prototype image.
        red_images: List of NumPy arrays representing the red channel of images.
        green_images: List of NumPy arrays representing the green channel of images.
        blue_images: List of NumPy arrays representing the blue channel of images.
        hue_images: List of NumPy arrays representing the hue channel of images.
        saturation_images: List of NumPy arrays representing the saturation channel of images.
        value_images: List of NumPy arrays representing the value (brightness) channel of images.
        name: string name of the class
        bins: Number of bins for the histogram (default: 256).

    Returns:
        Tuple containing the mean and variance of the RGB histograms.
    """
    # Lists to store histograms
    hist_reds, hist_greens, hist_blues = [], [], []
    hist_hues, hist_saturations, hist_values = [], [], []

    # Calculate histograms for each image of each channel
    for red, green, blue, hue, saturation, value in zip(
            red_images, green_images, blue_images, hue_images, saturation_images, value_images):
        hist_reds.append(np.histogram(red, bins=bins, range=(0, 256))[0])
        hist_greens.append(np.histogram(green, bins=bins, range=(0, 256))[0])
        hist_blues.append(np.histogram(blue, bins=bins, range=(0, 256))[0])
        hist_hues.append(np.histogram(hue, bins=bins, range=(0, 256))[0])
        hist_saturations.append(np.histogram(saturation, bins=bins, range=(0, 256))[0])
        hist_values.append(np.histogram(value, bins=bins, range=(0, 256))[0])

    # Calculate the mean of RGB histograms
    mean_hist_red = np.mean(hist_reds, axis=0)
    mean_hist_green = np.mean(hist_greens, axis=0)
    mean_hist_blue = np.mean(hist_blues, axis=0)

    # Calculate the mean of HSV histograms
    mean_hist_hue = np.mean(hist_hues, axis=0)
    mean_hist_saturation = np.mean(hist_saturations, axis=0)
    mean_hist_value = np.mean(hist_values, axis=0)

    # Calculate the variance of RGB histograms
    var_hist_red = np.var(hist_reds, axis=0)
    var_hist_green = np.var(hist_greens, axis=0)
    var_hist_blue = np.var(hist_blues, axis=0)

    # Create plots: 1 row and 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))

    # Plot the average prototype image
    axes[0].imshow(average_prototype)
    axes[0].set_title("Average Prototype Image")
    axes[0].axis("off")

    # Plot mean histograms for RGB channels
    axes[1].plot(mean_hist_red, color="red", label="Mean Histogram - Red")
    axes[1].plot(mean_hist_green, color="green", label="Mean Histogram - Green")
    axes[1].plot(mean_hist_blue, color="blue", label="Mean Histogram - Blue")
    axes[1].set_title("Mean Histogram - RGB Channels")
    axes[1].set_xlabel("Pixel Intensity")
    axes[1].set_ylabel("Average Frequency")
    axes[1].legend()

    # Plot variance for RGB channels
    axes[2].plot(var_hist_red, color="red", label="Variance - Red")
    axes[2].plot(var_hist_green, color="green", label="Variance - Green")
    axes[2].plot(var_hist_blue, color="blue", label="Variance - Blue")
    axes[2].set_title("Variance - RGB Channels")
    axes[2].set_xlabel("Pixel Intensity")
    axes[2].set_ylabel("Variance")
    axes[2].legend()

    # Plot mean histograms for HSV channels
    axes[3].plot(mean_hist_hue, color="black", label="Mean Histogram - Hue")
    axes[3].plot(mean_hist_saturation, color="blue", label="Mean Histogram - Saturation")
    axes[3].plot(mean_hist_value, color="gold", label="Mean Histogram - Value")
    axes[3].set_title("Mean Histogram - HSV Channels")
    axes[3].set_xlabel("Pixel Intensity")
    axes[3].set_ylabel("Average Frequency")
    axes[3].legend()

    # Display plots
    plt.tight_layout()
    plt.savefig(f"/home/mab0205/UTFPR/GitHub/DIP-Candy-Classification/docs/{name}average_histogram_variance.png", dpi=300, bbox_inches="tight")


def load_images(folder_path: str):
    """
    Load all images from a folder and convert them to NumPy arrays in RGB -> HSV format.

    Args:
        folder_path: Path to the folder where images are located.

    Returns:
        Tuple of lists containing the red, green, blue, hue, saturation, and value channels of the images.
    """
    red_images, green_images, blue_images = [], [], []
    hue_images, saturation_images, value_images = [], [], []

    for file in os.listdir(folder_path):
        if file.endswith((".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, file)
            img_bgr = cv2.imread(image_path)

            if img_bgr is not None:
                # Split into RGB channels
                blue_channel, green_channel, red_channel = cv2.split(img_bgr)
                red_images.append(red_channel)
                green_images.append(green_channel)
                blue_images.append(blue_channel)

                # Convert to HSV and split channels
                img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2HSV)
                hue_channel, saturation_channel, value_channel = cv2.split(img_hsv)
                hue_images.append(hue_channel)
                saturation_images.append(saturation_channel)
                value_images.append(value_channel)
            else:
                print(f"Failed to load image: {file}")
        else:
            print(f"Skipped non-image file: {file}")

    return red_images, green_images, blue_images, hue_images, saturation_images, value_images



def generate_image_statitics(folder_path: str):

    # List all subdirectories (classes) in the folder_path
    class_folders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    
    for class_folder in class_folders:
        class_name = os.path.basename(class_folder)
        print(class_name)
        print(class_folder)
        red_images, green_images, blue_images, hue_images, saturation_images, value_images = load_images(class_folder)

        prototype = average_prototype(red_images, green_images, blue_images)


        histogram_mean_and_variance(prototype, red_images, green_images, blue_images,
                                                    hue_images, saturation_images, value_images, class_name)

        print(f"Class '{class_name}' statistics:")
        print("Average prototype calculated successfully!")
        print("Histogram variance calculated!")
        print("Mean histogram calculated!")