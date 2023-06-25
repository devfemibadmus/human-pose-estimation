import tensorflow as tf
import cv2
import threading
import numpy as np
from evaluation import *
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg

# Load the model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = input_details[0]['shape'][1]

# Function to process frames
def process_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame
        frame_resized = cv2.resize(frame, (input_size, input_size))
        input_image = np.expand_dims(frame_resized, axis=0)
        
        # Run model inference
        interpreter.set_tensor(input_details[0]['index'], input_image)
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
        
        # Visualize keypoints on frame
        output_frame = draw_prediction_on_image(frame, keypoints_with_scores)
        
        # Resize output frame to fit the display window
        display_frame = cv2.resize(output_frame, (0, 0), fx=0.5, fy=0.5)
        
        # Display the frame
        cv2.imshow('Output', display_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Start processing frames in a separate thread
thread = threading.Thread(target=process_frames)
thread.start()
