# Human Pose Estimation Project

This project focuses on Human Pose Estimation using the MoveNet model with TensorFlow Lite. The goal is to detect keypoint positions on a person's body in images and live video frames. The project provides a Flask web application for both image and live video input, showcasing the real-time capabilities of the model.

## Project Structure

As a Python enthusiast, understanding the structure and technologies involved in this project can provide insights into how AI is integrated into real-world applications.

### 1. TensorFlow Lite and MoveNet:

**TensorFlow Lite (TFLite):**
- TensorFlow is an open-source machine learning library, and TensorFlow Lite is a lightweight version designed for mobile and edge devices.
- TFLite enables the deployment of machine learning models on resource-constrained devices.

**MoveNet:**
- MoveNet is a lightweight and efficient model for human pose estimation, developed by Google.
- It predicts keypoint locations on a person's body, providing information about the positions of various body parts (e.g., nose, eyes, shoulders) in an image.

**Integration:**
- The project utilizes the TFLite interpreter to load and run the MoveNet model.
- The `evaluation.py` module contains functions to process model outputs and visualize keypoint predictions.

### 2. Flask Web Application:

**Flask:**
- Flask is a web framework for building web applications in Python.
- It is lightweight, easy to use, and facilitates the creation of web services.

**Web Routes:**
- The Flask application (`app.py`) defines routes for handling different functionalities.
- The root route (`/`) handles both image and live video input, processing frames, and rendering the results.
- Other routes, such as `/live` and `/video_feed`, provide live video streaming functionality.

**HTML Templates:**
- The project uses HTML templates (located in the `template/` directory) to structure the web pages rendered by Flask.
- Templates include `index.html` for image processing and `live.html` for live video streaming.

**File Upload:**
- The application uses Flask's file upload feature to handle user-uploaded images.
- Uploaded images are saved in the `medias/` directory.

### 3. Image Processing:

**Image Preprocessing:**
- The `process_image` function in `app.py` loads, resizes, and preprocesses the input image for model inference.

**Model Inference:**
- The pre-trained MoveNet model is used to perform keypoint detection on the input image.

**Visualization:**
- The `draw_prediction_on_image` function in `evaluation.py` visualizes the keypoint predictions on the original image.

**Result Display:**
- The results (original and predicted images) are displayed on the web interface.

### 4. Live Video Processing:

**Video Feed:**
- The `/video_feed` route streams live video frames captured from the camera.

**Frame Processing:**
- The `process_frames` function captures frames, preprocesses them, runs model inference, and streams the processed frames in real-time.

### 5. Project Structure:

**Model File:**
- The pre-trained MoveNet model is stored in the `model.tflite` file.

**Static Files:**
- Uploaded and processed images are stored in the `medias/` directory.
- Static files (CSS, JS) and HTML templates are organized within the project structure.

### 6. Requirements:

Dependencies are listed in the `requirements.txt` file, and they can be installed using pip.
`Python >=3.0, <=3.11.5`

### 7. Sample Results:

Sample results, such as the original image and the corresponding result image showcase the model's performance, these are my babe picture, she's beautiful yes i know thank you.

- Original Image: `zainab.png`
![Original Image](medias/zainab.JPG)

- Result Image: `zainab_result.jpg`
![Result Image](medias/zainab_result.JPG?raw=true)

### 8. Acknowledgments:

The project acknowledges the use of the MoveNet model and TensorFlow Lite for efficient and real-time human pose estimation.

Feel free to explore and enhance this project for your specific use cases or integrate it into other applications.

If you encounter any issues or have suggestions for improvements, please don't hesitate to reach out or contribute to the project.

Enjoy exploring Human Pose Estimation with MoveNet!

### 9. Next Steps:

Python enthusiasts can explore and enhance the project, potentially integrating it into other applications or experimenting with different models. Understanding this structure provides insights into how Python, TensorFlow Lite, and Flask can be combined to create a practical AI application. Feel free to explore further, experiment with different models, or contribute to the project for continuous improvement!



