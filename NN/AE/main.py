from train import train_model
from evaluate import evaluate_model
from load_map import split_and_save_datasets

if __name__ == "__main__":
    # Parameters
    #data_dir = "c:/Users/yigit/Desktop/EE493/NN/AE/range_angle_maps"  # Path to RA map files, maybe we can change this approach but let it stay for now
    data_dir = "c:/Users/yigit/Desktop/RA_NPY/0128"
    base_dir = "c:/Users/yigit/Desktop/EE493/NN/AE/datasets"  # Base directory with 'train' and 'val_images' subfolders
    model_save_path = "saved_models/autoencoder_model.keras"
    training_metrics_path = "results/training_metrics.txt"
    evaluation_metrics_path = "results/evaluation_metrics.txt"
    input_shape = (128, 128, 1)  # Example input shape, adjust based on the specific data!!!!
    epochs = 10 #we will modify these in case of underfitting/overfitting
    batch_size = 16
    
    split_and_save_datasets(data_dir, base_dir)
    
    #Train the model
    train_model(base_dir, model_save_path, input_shape, epochs, batch_size, training_metrics_path)
    
    #Evaluate the model
    evaluate_model(base_dir, model_save_path, data_dir, evaluation_metrics_path)