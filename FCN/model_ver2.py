import tensorflow as tf
from tensorflow.keras import layers, models
from utils import weighted_mse, f1_score

def build_fcn_32s(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder (Downsampling)
    x1 = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.LeakyReLU(alpha=0.1)(x1)  # LeakyReLU instead of Clipped ReLU
    x = layers.MaxPooling2D((2, 3), padding='same')(x1)
    
    x2 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.LeakyReLU(alpha=0.1)(x2)
    x = layers.MaxPooling2D((2, 3), padding='same')(x2)
    
    x3 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=2)(x)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.LeakyReLU(alpha=0.1)(x3)
    
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=2)(x3)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.3)(x)  # Regularization
    
    # Decoder (Upsampling)
    x = layers.Conv2DTranspose(128, (3, 3), padding='same', kernel_initializer='he_normal', strides=(2, 3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    # Fix: Use UpSampling2D to align spatial dimensions for concatenation
    x3_up = layers.UpSampling2D(size=(2, 3))(x3)  # Upsample x3 to match spatial dimensions of x
    x = layers.Concatenate()([x, x3_up])  # Concatenate after upsampling x3

    x = layers.Conv2DTranspose(64, (3, 3), padding='same', kernel_initializer='he_normal', strides=(2, 3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    x2_up = layers.UpSampling2D(size=(2, 3))(x2)  # Upsample x2 to match spatial dimensions of x
    x = layers.Concatenate()([x, x2_up])

    x = layers.Conv2DTranspose(32, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    #x1_up = layers.UpSampling2D(size=(2, 3))(x1)  # Upsample x1 to match spatial dimensions of x
    x = layers.Concatenate()([x, x1])
    
    # Output Layer
    decoder_output = layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Model
    model = models.Model(inputs, decoder_output)
    return model

# Define model
input_shape = (128, 9, 1)  # Adjusted shape
model = build_fcn_32s(input_shape)
model.compile(optimizer='adam', loss=weighted_mse, metrics=['accuracy', f1_score])

# Print model summary
model.summary()