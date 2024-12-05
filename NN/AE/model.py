import tensorflow as tf
from tensorflow.keras import layers, models
from utils import weighted_mse

def build_autoencoder(input_shape):
    """
    Build and return the autoencoder model.

    Parameters:
        input_shape (tuple): Shape of the input data (e.g., (64, 64, 1)).

    Returns:
        autoencoder (Model): Compiled autoencoder model.
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

    # Decoder
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(encoder_output)  # 64x64x32
    x = layers.UpSampling2D((2, 2))(x)  # 128x128x32
    x = layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)  # 128x128x16
    x = layers.ReLU(max_value=1.0)(x)  # Clipped ReLU
    x = layers.add([x, x2])  # Skip connection with x2 (128x128x16)
    x = layers.UpSampling2D((2, 2))(x)  # 256x256x16
    x = layers.Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(x)  # 256x256x8
    x = layers.ReLU(max_value=1.0)(x)  # Clipped ReLU
    x = layers.add([x, x1])  # Skip connection with x1 (256x256x8)
    decoder_output = layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)  # 256x256x1

    # Autoencoder: Combine encoder and decoder
    autoencoder = tf.keras.Model(encoder_input, decoder_output, name="autoencoder")
    autoencoder.compile(optimizer='adam', loss=weighted_mse)

    
    return autoencoder

print("Success")