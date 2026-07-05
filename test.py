import os
import tensorflow as tf

# Load your trained weights
autoencoder = tf.keras.models.load_model('models/autoencoder.h5')
classifier = tf.keras.models.load_model('models/classifier.h5')

# Install specific light-weight conversion tools
# pip install tf2onnx onnxruntime

import tf2onnx

# Convert and save models cleanly
spec_ae = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)
tf2onnx.convert.from_keras(autoencoder, input_signature=spec_ae, output_path="models/autoencoder.onnx")

spec_cl = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)
tf2onnx.convert.from_keras(classifier, input_signature=spec_cl, output_path="models/classifier.onnx")

print("ONNX Models successfully generated!")