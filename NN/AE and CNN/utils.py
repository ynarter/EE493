import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from keras.saving import register_keras_serializable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

@register_keras_serializable()
def weighted_mse(y_true, y_pred):
    weight = K.cast(K.greater(y_true, 0.7), "float32") * 15.0 + 1.0
    return K.mean(weight * K.square(y_true - y_pred))

def calculate_rmse(original, reconstructed):
    """
    Calculate the Root Mean Square Error (RMSE) between the original and reconstructed data.
    
    Parameters:
        original (numpy array): Ground truth data.
        reconstructed (numpy array): Reconstructed data from the model.
    
    Returns:
        float: RMSE value.
    """
    return np.sqrt(mean_squared_error(original.flatten(), reconstructed.flatten()))

def calculate_snr(original, reconstructed):
    """
    Calculate the Signal-to-Noise Ratio (SNR) for reconstructed data.
    
    Parameters:
        original (numpy array): Ground truth data.
        reconstructed (numpy array): Reconstructed data from the model.
    
    Returns:
        float: SNR value in decibels.
    """
    signal_power = np.sum(original**2)
    noise_power = np.sum((original - reconstructed)**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_correlation(original, reconstructed):
    """
    Calculate the 2D correlation coefficient between the original and reconstructed data.
    
    Parameters:
        original (numpy array): Ground truth data.
        reconstructed (numpy array): Reconstructed data from the model.
    
    Returns:
        float: Correlation coefficient value.
    """
    original_flat = original.flatten()
    reconstructed_flat = reconstructed.flatten()
    correlation = np.corrcoef(original_flat, reconstructed_flat)[0, 1]
    return correlation

def visualize_results(original, reconstructed, num_images=5):
    """
    Visualize a few examples of original vs reconstructed images side by side.
    
    Parameters:
        original (numpy array): Ground truth data.
        reconstructed (numpy array): Reconstructed data from the model.
        num_images (int): Number of examples to display.
    """
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        # Original
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original[i+20].squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Reconstructed
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructed[i+20].squeeze(), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def add_gaussian_noise(data, mean=0.0, std_dev=0.05):
    """
    Add Gaussian noise to the data.
    
    Parameters:
        data (numpy array): Input data to which noise will be added.
        mean (float): Mean of the Gaussian noise.
        std_dev (float): Standard deviation of the Gaussian noise.
    
    Returns:
        numpy array: Data with added Gaussian noise.
    """
    noise = np.random.normal(mean, std_dev, data.shape)
    return data + noise

def save_metrics_to_file(metrics, file_path="metrics.txt"):
    """
    Save performance metrics to a text file.
    
    Parameters:
        metrics (dict): Dictionary containing metric names and values.
        file_path (str): Path to save the metrics file.
    """
    with open(file_path, 'w') as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")
    print(f"Metrics saved to {file_path}")

def visualize_predictions(test_input, test_target, predicted_labels, n_samples=10):
    """
    Visualize a few predictions along with their ground truth labels.

    Parameters:
        test_input (numpy.ndarray): Test input images.
        test_target (numpy.ndarray): Ground truth binary labels.
        predicted_labels (numpy.ndarray): Predicted binary labels.
        n_samples (int): Number of samples to display.
    """
    indices = np.random.choice(len(test_input), n_samples, replace=False)
    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(test_input[idx].squeeze(), cmap="gray")  # Assuming grayscale images
        plt.title(f"True: {test_target[idx][0]}, Pred: {predicted_labels[idx]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    
def display_metrics(test_target, predictions, predicted_labels):
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
