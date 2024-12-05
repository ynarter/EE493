import numpy as np
from tensorflow.keras.models import load_model
from load_data import load_and_preprocess_data
from utils import weighted_mse, calculate_rmse, calculate_snr, calculate_correlation, visualize_results, save_metrics_to_file

def evaluate_model(model_path, data_dir, metrics_path="results/evaluation_metrics.txt"):
    """
    Load a trained model and evaluate it on the test dataset.

    Parameters:
        model_path (str): Path to the saved model.
        data_dir (str): Directory containing RA map files.
        metrics_path (str): Path to save evaluation metrics.
    """
    #Load data
    _, _, _, _, test_input, test_target = load_and_preprocess_data(data_dir)
    
    #Load model
    autoencoder = load_model(model_path)
    
    #Evaluate according to our metrics
    reconstructed = autoencoder.predict(test_input)
    rmse = calculate_rmse(test_target, reconstructed)
    snr = calculate_snr(test_target, reconstructed)
    correlation = calculate_correlation(test_target, reconstructed)
    
    #Print metrics
    print(f"RMSE: {rmse}")
    print(f"SNR (dB): {snr}")
    print(f"Correlation: {correlation}")
    
    #Save metrics
    metrics = {"RMSE": rmse, "SNR (dB)": snr, "Correlation": correlation}
    #save_metrics_to_file(metrics, metrics_path)
    
    #Visualize results (in utils)!!!!
    visualize_results(test_target, reconstructed)


print("Success 3")