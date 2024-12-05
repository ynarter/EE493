import os
from load_data import load_and_preprocess_data
from model import build_autoencoder
from utils import save_metrics_to_file

def train_model(data_dir, model_save_path, input_shape, epochs=50, batch_size=16, metrics_path="results/training_metrics.txt"):
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
    train_input, train_target, val_input, val_target, test_input , test_target = load_and_preprocess_data(data_dir)
    
    #Build the model using the code
    autoencoder = build_autoencoder(input_shape)
    #autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
    
    #Train the model
    history = autoencoder.fit(
        train_input, train_target,
        validation_data=(val_input, val_target),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True
    )
    
    #Save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    autoencoder.save(model_save_path)
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