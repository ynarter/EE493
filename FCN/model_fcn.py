import tensorflow as tf
from tensorflow.keras import layers, models
from utils import weighted_mse, f1_score

def build_fcn_32s(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder (Downsampling)
    x1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)  # (128x9x32)
    x1 = layers.ReLU(max_value=6.0)(x1)  # Clipped ReLU
    x = layers.MaxPooling2D((2, 3), padding='same')(x1)  # (64x9x32)
    
    x2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)  # (64x9x64)
    x2 = layers.ReLU(max_value=6.0)(x2)  # Clipped ReLU
    x = layers.MaxPooling2D((2, 3), padding='same')(x2)  # (32x9x64)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)  # (32x9x128)
    x = layers.ReLU(max_value=6.0)(x)  # Clipped ReLU
    
    # Further Convolutional Layer
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)  # (32x9x128)
    x = layers.ReLU(max_value=6.0)(x)  # Clipped ReLU
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)  # (32x9x256)
    x = layers.ReLU(max_value=6.0)(x)  # Clipped ReLU
    
    # Decoder (Upsampling)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)  # (32x9x64)
    x = layers.UpSampling2D((2, 3))(x)  # (64x9x64)
    x = layers.add([x, x2])  # Skip connection (64x9x64)
    
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)  # (64x9x32)
    x = layers.UpSampling2D((2, 3))(x)  # (128x9x32)
    x = layers.add([x, x1])  # Skip connection (128x9x32)
    
    # Output layer
    decoder_output = layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)  # (128x9x1)
    
    # Model
    model = models.Model(inputs, decoder_output)
    return model

# Define model
input_shape = (128, 9, 1)  # Adjusted shape
model = build_fcn_32s(input_shape)
model.compile(optimizer='adam', loss=weighted_mse, metrics=['accuracy', f1_score])

# Print model summary
model.summary()