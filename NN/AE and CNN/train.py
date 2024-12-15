import os
import numpy as np
from load_data import load_and_preprocess_data
from model import build_autoencoder
from model_ae_cnn import build_ae_cnn
from utils import save_metrics_to_file
from load_map import load_images_from_folder, load_labels

def train_model(base_dir, model_save_path, input_shape, epochs=50, batch_size=16, metrics_path="results/training_metrics.txt"):
    """
    Train the autoencoder and save the trained model.
    
    Parameters:
        data_dir (str): Directory containing RA map files.
        model_save_path (str): Path to save the trained model.
        input_shape (tuple): Shape of the input data (e.g., (64, 64, 1)).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        metrics_path (str): Path to save training metrics.
    """
    #Load and preprocess data using our code
    #train_input, train_target, val_input, val_target, test_input , test_target = load_and_preprocess_data(data_dir)
    
    train_input_dir = os.path.join(base_dir, "train_input")
    val_input_dir = os.path.join(base_dir, "val_input")
    
    train_target_dir = os.path.join(base_dir, "train_labels")
    val_target_dir = os.path.join(base_dir, "val_labels")
    
    train_input = load_images_from_folder(train_input_dir, image_size=input_shape[:2])
    val_input = load_images_from_folder(val_input_dir, image_size=input_shape[:2])
    
    train_target = load_labels(train_target_dir)
    val_target = load_labels(val_target_dir)
    
    #train_target = load_images_from_folder(train_target_dir, image_size=input_shape[:2])
    #val_target = load_images_from_folder(val_target_dir, image_size=input_shape[:2])
    
    #train_target = np.squeeze(train_target)
    #val_target = np.squeeze(val_target)
    
    print(f"Training input shape: {train_input.shape}, Training target shape: {train_target.shape}")
    print(f"Validation input shape: {val_input.shape}, Validation target shape: {val_target.shape}")
    print("Data loaded and preprocessed successfully.")
    
    #Build the model using the code
    _, ae_cnn = build_ae_cnn(input_shape)
    #autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
    
    #Train the model
    history = ae_cnn.fit(
        train_input, train_target,
        validation_data=(val_input, val_target),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True
    )
    
    #Save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    ae_cnn.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    #Save training metrics like train loss, validation loss etc., we can specify other metrics for our purposes
    metrics = {
        "Final Training Loss": history.history['loss'][-1],
        "Final Validation Loss": history.history['val_loss'][-1],
        "Epochs": epochs,
        "Batch Size": batch_size
    }
    #save_metrics_to_file(metrics, metrics_path)


print("Success 2")