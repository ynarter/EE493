import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from load_data import load_and_preprocess_data
from load_map import load_images_from_folder, load_labels
import os
from utils import weighted_mse, calculate_rmse, calculate_snr, calculate_correlation, visualize_results, save_metrics_to_file
from utils import visualize_predictions, display_metrics

def evaluate_model(base_dir, model_path, data_dir, metrics_path="results/evaluation_metrics.txt"):
    """
    Load a trained model and evaluate it on the test dataset.

    Parameters:
        model_path (str): Path to the saved model.
        data_dir (str): Directory containing RA map files.
        metrics_path (str): Path to save evaluation metrics.
    """
    #Load data
    
    test_input_dir = os.path.join(base_dir, "test_input")
    test_target_dir = os.path.join(base_dir, "test_labels")
    
    test_input = load_images_from_folder(test_input_dir)
    test_target = load_labels(test_target_dir)
    
    print(f"Testing input shape: {test_input.shape}, Testing target shape: {test_target.shape}")
    print("Data loaded and preprocessed successfully.")
    
    #Load model
    ae_cnn = load_model(model_path)
    
    #Evaluate according to our metrics
    predictions = ae_cnn.predict(test_input)
    predicted_labels = (predictions > 0.5).astype(int).flatten()
    
    """
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
    """
    
    display_metrics(test_target, predictions, predicted_labels)
    
    """
    accuracy = accuracy_score(test_target, predicted_labels)
    precision = precision_score(test_target, predicted_labels)
    recall = recall_score(test_target, predicted_labels)
    f1 = f1_score(test_target, predicted_labels)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


    # Plot ROC curve
    fpr, tpr, _ = roc_curve(test_target, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    """
    
    visualize_predictions(test_input, test_target, predicted_labels)

    
    


print("Success 3")