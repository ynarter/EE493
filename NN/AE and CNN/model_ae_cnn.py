import tensorflow as tf
from tensorflow.keras import layers, models
from utils import weighted_mse

def build_ae_cnn(input_shape):
    """
    Build and return the autoencoder model with CNN for binary classification.
    The decoder part has been removed, and more layers are added to the CNN.

    Parameters:
        input_shape (tuple): Shape of the input data (e.g., (64, 64, 1)).

    Returns:
        autoencoder (Model): Compiled autoencoder model for reconstruction.
        classifier (Model): Model for binary classification.
    """
    # Encoder
    encoder_input = tf.keras.Input(shape=input_shape, name="encoder_input")
    x1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoder_input)  # 256x256x8
    x1 = layers.ReLU(max_value=1.0)(x1)  # Clipped ReLU
    x = layers.MaxPooling2D((2, 2), padding='same')(x1)  # 128x128x8
    x2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)  # 128x128x16
    x2 = layers.ReLU(max_value=1.0)(x2)  # Clipped ReLU
    x = layers.MaxPooling2D((2, 2), padding='same')(x2)  # 64x64x16
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # 64x64x32
    x = layers.ReLU(max_value=1.0)(x)  # Clipped ReLU
    encoder_output = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)  # 64x64x64
    encoder_output = layers.ReLU(max_value=1.0)(encoder_output)  # Clipped ReLU

    #only the encoder for feature extraction (no decoder part)
    autoencoder = tf.keras.Model(encoder_input, encoder_output, name="autoencoder")
    autoencoder.compile(optimizer='adam', loss=weighted_mse)

    #CNN for binary classification (added more layers)
    flatten = layers.Flatten()(encoder_output)  # Flatten the encoder output (latent space)
    x = layers.Dense(128, activation='relu')(flatten)  # Fully connected layer
    x = layers.Dropout(0.5)(x)  # Dropout for regularization
    x = layers.Dense(256, activation='relu')(x)  #more layers
    x = layers.Dropout(0.5)(x)  # Dropout layer for regularization
    x = layers.Dense(512, activation='relu')(x)  #more layers
    x = layers.Dropout(0.5)(x)  # Dropout layer for regularization
    classification_output = layers.Dense(1, activation='sigmoid', name="classification_output")(x)  # Binary classification output

    # Classification model
    classifier = tf.keras.Model(encoder_input, classification_output, name="classifier")
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return autoencoder, classifier

print("Success")
