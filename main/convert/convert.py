import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.keras.models import model_from_json

# Load Keras model
with open('hpe_hourglass_stacks_04_.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('hpe_epoch107_.hdf5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Convert the TFLite model to JavaScript format
tfjs.converters.save_tfjs_model(converter.model, 'model_js')
