from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf, cv2, sys, os
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

if __name__ == '__main__':
    app.run(debug=True)
