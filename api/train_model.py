import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

os.makedirs('models', exist_ok=True)

def train_and_save_models():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize and reshape
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # Save a subset of test data for backend fast random fetching
    np.save('models/x_test.npy', x_test)
    np.save('models/y_test.npy', y_test)

    # 1. Train Classifier (For Confidence Score)
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
    classifier.save('models/classifier.h5')

    # 2. Train Autoencoder
    print("Training Autoencoder...")
    noise_factor = 0.4
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    # Encoder
    input_img = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x) # Latent Representation

    # Decoder
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(x_train_noisy, x_train, epochs=3, batch_size=128, validation_data=(x_test_noisy, x_test), verbose=1)
    autoencoder.save('models/autoencoder.h5')
    print("All models successfully trained and saved!")

if __name__ == '__main__':
    train_and_save_models()