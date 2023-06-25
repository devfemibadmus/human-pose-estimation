# you can run this to estimate pose on images, e.g python media_pose.py path-to-image.format
import tensorflow as tf
import cv2
from evaluation import *
import sys
import os

def process_image(image_path):
    # Load the input image
    image = cv2.imread(image_path)

    # Get the original image size
    original_height, original_width, _ = image.shape

    # Resize and pad the image to the model's input size
    input_image = cv2.resize(image, (input_size, input_size))
    input_image = np.expand_dims(input_image, axis=0)

    # Run model inference
    keypoints_with_scores = movenet(input_image)

    # Visualize the predictions with the original image
    output_overlay = draw_prediction_on_image(image, keypoints_with_scores)

    # Save the result with modified image name and format
    result_image_name = image_path.split('.')[0] + '_result.' + image_path.split('.')[1]
    cv2.imwrite(result_image_name, output_overlay)

    print("Result image saved as:", result_image_name)

# Check if image path is provided as a command-line argument
if len(sys.argv) != 2:
    print("Please provide the image path as a command-line argument.")
else:
    # Get the image path from command-line argument
    image_path = sys.argv[1]
    process_image(image_path)
