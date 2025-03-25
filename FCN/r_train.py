import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from load import load_data_from_folder
#from model_fcn import model
from model_ver2 import model
from tensorflow.keras.callbacks import LearningRateScheduler
import os
from utils import weighted_mse, f1_score

DATASET_PATH = "c:/Users/yigit/Desktop/pixelwise_final_2/pixelwise_final"
# Load train, test, and validation sets
X_train, y_train_masks, _ = load_data_from_folder(os.path.join(DATASET_PATH, "train/x_data"), os.path.join(DATASET_PATH, "train/y_data"))
X_val, y_val_masks, _ = load_data_from_folder(os.path.join(DATASET_PATH, "val/x_data"), os.path.join(DATASET_PATH, "val/y_data"))

# Reshape for CNN input (Add a channel dimension)
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
y_train_masks = y_train_masks[..., np.newaxis]  # Add channel dimension
y_val_masks = y_val_masks[..., np.newaxis]

# Print dataset shapes for debugging
print(f"Training Data Shape: {X_train.shape}, Masks Shape: {y_train_masks.shape}")
print(f"Validation Data Shape: {X_val.shape}, Masks Shape: {y_val_masks.shape}")


# Training Hyperparameters
EPOCHS = 50
BATCH_SIZE = 32

# === Custom Callback to Show Training Progress ===
class TrainingProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}/{EPOCHS} - "
              f"Loss: {logs['loss']:.4f}, "
              f"Classification Accuracy: {logs.get('classification_output_accuracy', 0):.4f}, "
              f"Val Loss: {logs['val_loss']:.4f}, "
              f"Val Classification Accuracy: {logs.get('val_classification_output_accuracy', 0):.4f}")

# Define Learning Rate Scheduler
def lr_schedule(epoch):
    initial_lr = 0.001
    drop_factor = 0.5
    drop_every = 20
    return initial_lr * (drop_factor ** (epoch // drop_every))

lr_scheduler = LearningRateScheduler(lr_schedule)

# === Train the Model ===
history = model.fit(
    X_train, 
    y_train_masks,
    validation_data=(X_val, y_val_masks),
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE,
)

# Save the trained model
model.save("model_50_32_improved.keras")
print("Model saved successfully.")

""" # === Training & Validation Accuracy Plot ===
plt.figure(figsize=(10, 4))
plt.plot(history.history.get('classification_output_accuracy', []), label='Train Accuracy')
plt.plot(history.history.get('val_classification_output_accuracy', []), label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Classification Accuracy Over Training')
plt.legend()
plt.grid(True)
plt.show() """
