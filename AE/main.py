from train import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    # Parameters
    data_dir = "data/"  # Path to RA map files
    model_save_path = "saved_models/autoencoder_model.h5"
    training_metrics_path = "results/training_metrics.txt"
    evaluation_metrics_path = "results/evaluation_metrics.txt"
    input_shape = (64, 64, 1)  # Example input shape, adjust based on your data
    epochs = 50
    batch_size = 16
    
    # Train the model
    train_model(data_dir, model_save_path, input_shape, epochs, batch_size, training_metrics_path)
    
    # Evaluate the model
    evaluate_model(model_save_path, data_dir, evaluation_metrics_path)

