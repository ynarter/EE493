import numpy as np
import matplotlib.pyplot as plt

array = np.load("c:/Users/yigit/Desktop/EE493/NN/AE/000000.npy")
print(f"Shape of the loaded array: {array.shape}")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Create a side-by-side plot

# Plot each channel
axes[0].imshow(array[:, :, 0], cmap='viridis')  # Channel 0
axes[0].set_title("Channel 0")
axes[0].axis('off')  # Remove axis ticks

axes[1].imshow(array[:, :, 1], cmap='viridis')  # Channel 1
axes[1].set_title("Channel 1")
axes[1].axis('off')  # Remove axis ticks

# Display the plots
plt.tight_layout()
plt.show()