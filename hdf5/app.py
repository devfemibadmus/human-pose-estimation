from flask import Flask, render_template, request, send_file, send_from_directory
from werkzeug.utils import secure_filename
from helper import AppHelper
from PIL import Image
import io, os, cv2
import numpy as np

app = Flask(__name__, template_folder='template')
app_helper = None
UPLOAD_FOLDER = 'medias/uploads'
RESULT_FOLDER = 'medias/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def load_and_preprocess_img(img_path):
    img = Image.open(img_path).convert('RGB')
    return np.array(img)


def load_model():
    global app_helper
    model_json = 'lib/models/result/hpe_hourglass_stacks_04_.json'
    model_weights = 'lib/models/result/hpe_epoch73_.hdf5'
    app_helper = AppHelper(model_weights=model_weights, model_json=model_json)
    return app_helper
    
def predict_pose(img_path):
    handle = load_model()

    scatter = handle.predict_in_memory(img_path, visualize_scatter=True, visualize_skeleton=False)
    skeleton = handle.predict_in_memory(img_path, visualize_scatter=True, visualize_skeleton=True)

    scatter_img = Image.fromarray(scatter)
    skeleton_img = Image.fromarray(skeleton)

    # Save scatter and skeleton images with original size and quality
    scatter_filename = 'scatter.jpg'
    skeleton_filename = 'skeleton.jpg'
    scatter_path = os.path.join(app.config['RESULT_FOLDER'], scatter_filename)
    skeleton_path = os.path.join(app.config['RESULT_FOLDER'], skeleton_filename)
    scatter_img.save(scatter_path, quality=100)
    skeleton_img.save(skeleton_path, quality=100)

    return scatter_path, skeleton_path

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
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # Process the image with the app_helper
            scatter_path, skeleton_path = predict_pose(upload_path)

            # Render the result.html template with image paths
            return render_template('result.html', original_image=upload_path, scatter_image=scatter_path, skeleton_image=skeleton_path)

    return render_template('index.html')

@app.route('/medias/<path:filename>')
def serve_static(filename):
    return send_from_directory('medias', filename)

if __name__ == '__main__':
    app.run(debug=True)
