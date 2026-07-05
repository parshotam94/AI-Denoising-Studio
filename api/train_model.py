import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Force absolute path tracking relative to this script's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def train_save_and_convert():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    np.save(os.path.join(MODELS_DIR, 'x_test.npy'), x_test)
    np.save(os.path.join(MODELS_DIR, 'y_test.npy'), y_test)

    # 1. Train Classifier
    print("Training Classifier...")
    classifier = models.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    classifier.fit(x_train, y_train, epochs=2, batch_size=128, validation_split=0.1, verbose=1)
    
    h5_classifier_path = os.path.join(MODELS_DIR, 'classifier.h5')
    classifier.save(h5_classifier_path)

    # 2. Train Autoencoder
    print("Training Autoencoder...")
    noise_factor = 0.4
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)

    input_img = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(x_train_noisy, x_train, epochs=3, batch_size=128, verbose=1)
    
    h5_ae_path = os.path.join(MODELS_DIR, 'autoencoder.h5')
    autoencoder.save(h5_ae_path)

    # 3. Convert directly to ONNX Format
    print("Converting models to ONNX format...")
    try:
        import tf2onnx
        
        spec_ae = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)
        tf2onnx.convert.from_keras(autoencoder, input_signature=spec_ae, output_path=os.path.join(MODELS_DIR, "autoencoder.onnx"))

        spec_cl = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)
        tf2onnx.convert.from_keras(classifier, input_signature=spec_cl, output_path=os.path.join(MODELS_DIR, "classifier.onnx"))
        
        print("🎉 Success! All .h5 and .onnx models are compiled and saved in the models/ directory.")
    except ImportError:
        print("❌ Error: Please install 'tf2onnx' and 'onnx' locally to generate the serverless weights files.")

if __name__ == '__main__':
    train_save_and_convert()