import tensorflow as tf
from tensorflow.keras import layers, models

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
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder_output = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Decoder
    x = layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(encoder_output)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoder_output = layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Autoencoder
    autoencoder = models.Model(encoder_input, decoder_output, name="autoencoder")
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

print("Success")