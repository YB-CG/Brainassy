import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

def has_color(image):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for red, blue, and yellow
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create masks for each color
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine masks to detect red, blue, or yellow
    combined_mask = mask_red | mask_blue | mask_yellow

    # Check if there are any non-zero pixels in the combined mask
    return np.any(combined_mask > 0)

def crop_brain_contour(image, plot=False):
    # Check if the image has red, blue, or yellow colors
    if has_color(image):
        # Return None for images with red, blue, or yellow colors
        return None

    # Continue with the existing code for cropping the brain contour

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set intensity thresholds for the brain region
    lower_threshold = 45  # Adjust as needed
    upper_threshold = 255  # Adjust as needed

    # Threshold the image, then perform a series of erosions + dilations
    thresh = cv2.inRange(gray, lower_threshold, upper_threshold)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # If no contours are found, return None
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # Crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    return new_image

