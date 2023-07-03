from flask import Flask, render_template, Response, request, send_from_directory
import tensorflow as tf, cv2, threading, numpy as np, sys, os
from werkzeug.utils import secure_filename
from evaluation import *

app = Flask(__name__, template_folder='template')


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

    return result_image_name
    
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']

        # If user does not select a file, browser also
        # submits an empty part without filename
        if file.filename == '':
            return 'No selected file'

        if file:
            # Save the uploaded image file
            filename = secure_filename(file.filename)
            upload_path = os.path.join('medias', filename)
            file.save(upload_path)

            predict_image = process_image(upload_path)

            # Render the result.html template with image paths
            return render_template('result.html', original_image=upload_path, predict_image=predict_image)

    return render_template('index.html')

@app.route('/medias/<path:filename>')
def serve_static(filename):
    return send_from_directory('medias', filename)



# Load the model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = input_details[0]['shape'][1]

# Function to process frames
def process_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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
        
        # Convert frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', output_frame)
        
        # Yield the JPEG frame as a response to the client
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    
    cap.release()

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True)
