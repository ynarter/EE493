import numpy as np
from tensorflow.keras.models import load_model
from load_data import load_and_preprocess_data
from utils import calculate_rmse, calculate_snr, calculate_correlation, visualize_results, save_metrics_to_file

def evaluate_model(model_path, data_dir, metrics_path="results/evaluation_metrics.txt"):
    """
    Load a trained model and evaluate it on the test dataset.

    Parameters:
        model_path (str): Path to the saved model.
        data_dir (str): Directory containing RA map files.
        metrics_path (str): Path to save evaluation metrics.
    """
    # Load data
    _, _, test_data = load_and_preprocess_data(data_dir)
    
    # Load model
    autoencoder = load_model(model_path)
    
    # Evaluate
    reconstructed = autoencoder.predict(test_data)
    rmse = calculate_rmse(test_data, reconstructed)
    snr = calculate_snr(test_data, reconstructed)
    correlation = calculate_correlation(test_data, reconstructed)
    
    # Print metrics
    print(f"RMSE: {rmse}")
    print(f"SNR (dB): {snr}")
    print(f"Correlation: {correlation}")
    
    # Save metrics
    metrics = {"RMSE": rmse, "SNR (dB)": snr, "Correlation": correlation}
    save_metrics_to_file(metrics, metrics_path)
    
    # Visualize results
    visualize_results(test_data, reconstructed)


print("Success 3")