import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocess import has_color, crop_brain_contour

# Load an image
image_path = "/home/skay/Pictures/Wallpapers2.jpg"
image = cv2.imread(image_path)

# Check if the image has red, blue, or yellow colors
if has_color(image):
    print("Image has red, blue, or yellow colors. Skipping further processing.")
else:
    # Crop the brain contour
    cropped_image = crop_brain_contour(image)

    if cropped_image is not None:
        # Display the original and cropped images
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        plt.subplot(1, 2, 2)
        plt.title("Cropped Image")
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

        plt.show()
    else:
        print("No brain contour found in the image.")
