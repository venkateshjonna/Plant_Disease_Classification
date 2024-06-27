import cv2
import numpy as np

def adjust_hue(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] += value
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_saturation(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] += value
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_brightness(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] += value
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def extract_green_channel(image):
    green_channel = np.zeros_like(image)
    green_channel[:,:,1] = image[:,:,1]
    return green_channel

def double_image_size(image):
    return cv2.resize(image, (2*image.shape[1], 2*image.shape[0]))

# Load the image


try:
  image = cv2.imread('C://Users//venkatesh//Downloads//inst_profile.png')
except Exception as e:
  print(f"Error reading image: {e}")

# Perform color filtering operations
hue_adjusted = adjust_hue(image, 20)  # Adjust hue by 20 degrees
saturation_adjusted = adjust_saturation(image, 50)  # Increase saturation by 50
brightness_adjusted = adjust_brightness(image, 50)  # Increase brightness by 50
green_channel = extract_green_channel(image)
doubled_image = double_image_size(image)

# Display the results
cv2.imshow('Original', image)
cv2.imshow('Hue Adjusted', hue_adjusted)
cv2.imshow('Saturation Adjusted', saturation_adjusted)
cv2.imshow('Brightness Adjusted', brightness_adjusted)
cv2.imshow('Green Channel', green_channel)
cv2.imshow('Doubled Image', doubled_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
