import numpy as np
import matplotlib as plt
import random

# Function to visualize augmented images and their masks
def visualize_augmentations(dataset, num_samples=4):
    fig, axs = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))

    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        image, mask = dataset[idx]

        # Convert tensor to numpy array for visualization
        image_np = image.permute(1, 2, 0).cpu().numpy()
        mask_np = mask.cpu().numpy()

        # Plot image and masks
        axs[i, 0].imshow(image_np)
        axs[i, 0].set_title("Image")
        axs[i, 1].imshow(mask_np[..., 0], cmap='gray')
        axs[i, 1].set_title("Matrix Mask")
        axs[i, 2].imshow(mask_np[..., 1], cmap='gray')
        axs[i, 2].set_title("Secondary Mask")
        axs[i, 3].imshow(mask_np[..., 2], cmap='gray')
        axs[i, 3].set_title("Tertiary Mask")

        for j in range(4):
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.show()