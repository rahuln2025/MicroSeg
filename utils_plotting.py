
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches

# Function to visualize augmented images and their masks
def visualize_augmentations(dataset, num_samples=4, save_path = None):
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
    if save_path:
        plt.savefig(save_path, dpi = 300)
    
    #plt.show()


# Function to visualize predictions and true masks
def visualize_predictions(model, test_dataset, device, num_samples=4, save_path = None):
    model.eval()
    fig, axs = plt.subplots(2 * num_samples, 4, figsize=(20, 10 * num_samples))

    for i in range(num_samples):
        idx = random.randint(0, len(test_dataset) - 1)
        image, true_mask = test_dataset[idx]
        image = image.to(device).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            if output.shape[1] == 1:
                output = torch.sigmoid(output)
            else:
                output = torch.softmax(output, dim=1)
            output = output.squeeze().cpu().numpy()

        image = image.squeeze().permute(1, 2, 0).cpu().numpy()
        true_mask = true_mask.cpu().numpy()  # Ensure correct shape

        # Plot original image
        axs[2 * i, 0].imshow(image)
        axs[2 * i, 0].set_title("Image")
        
        # Plot true masks
        axs[2 * i, 1].imshow(true_mask[..., 0], cmap='gray')
        axs[2 * i, 1].set_title("True Matrix Mask")
        axs[2 * i, 2].imshow(true_mask[..., 1], cmap='gray')
        axs[2 * i, 2].set_title("True Secondary Mask")
        axs[2 * i, 3].imshow(true_mask[..., 2], cmap='gray')
        axs[2 * i, 3].set_title("True Tertiary Mask")
        
        # Plot predicted masks
        axs[2 * i + 1, 0].imshow(image)
        axs[2 * i + 1, 0].set_title("Image")
        axs[2 * i + 1, 1].imshow(output[0], cmap='gray')
        axs[2 * i + 1, 1].set_title("Predicted Matrix Mask")
        axs[2 * i + 1, 2].imshow(output[1], cmap='gray')
        axs[2 * i + 1, 2].set_title("Predicted Secondary Mask")
        axs[2 * i + 1, 3].imshow(output[2], cmap='gray')
        axs[2 * i + 1, 3].set_title("Predicted Tertiary Mask")

        for j in range(4):
            axs[2 * i, j].axis('off')
            axs[2 * i + 1, j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi = 300)
    #plt.show()


def visualize_prediction_accuracy_2(model, test_dataset, device, num_samples=4, save_path = None):
    model.eval()
    fig, axs = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))

    for i in range(num_samples):
        idx = random.randint(0, len(test_dataset) - 1)
        image, true_mask = test_dataset[idx]
        image = image.to(device).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            if output.shape[1] == 1:
                output = torch.sigmoid(output)
            else:
                output = torch.softmax(output, dim=1)
            output = output.squeeze().permute(1, 2, 0).cpu().numpy()
        
        
        image = image.squeeze().permute(1, 2, 0).cpu().numpy()
        true_mask = true_mask.cpu().numpy()
        
        #print("output: ", output.shape)
        #print("image: ", image.shape)
        #print("true mask: ", true_mask.shape)
        if output.shape != true_mask.shape:
            raise ValueError(f"Shape mismatch: Predicted mask shape {output.shape} and true mask shape {true_mask.shape} are incompatible for visualization.")

        axs[i, 0].imshow(image)
        axs[i, 0].set_title("Image")

        for j in range(3):
            pred_mask = output[:, :, j] > 0.5
            true_class_mask = true_mask[:, :, j]

            overlay = np.zeros((*true_class_mask.shape, 3), dtype=np.uint8)
            #print("pred_mask :", pred_mask.shape)
            #print("true_class_Mask: ", true_class_mask.shape)
            
            true_positives = (pred_mask == 1) & (true_class_mask == 1)
            true_negatives = (pred_mask == 0) & (true_class_mask == 0)
            false_positives = (pred_mask == 1) & (true_class_mask == 0)
            false_negatives = (pred_mask == 0) & (true_class_mask == 1)

            overlay[true_positives] = [255, 255, 255]  # White
            overlay[true_negatives] = [0, 0, 0]        # Black
            overlay[false_positives] = [0, 0, 255]     # Blue
            overlay[false_negatives] = [255, 105, 180] # Pink

            axs[i, j + 1].imshow(overlay)
            axs[i, j + 1].set_title(f"Mask {j+1} Overlay")

        for j in range(4):
            axs[i, j].axis('off')

    white_patch = mpatches.Patch(color='white', label='True Positive')
    black_patch = mpatches.Patch(color='black', label='True Negative')
    blue_patch = mpatches.Patch(color='blue', label='False Positive')
    pink_patch = mpatches.Patch(color='pink', label='False Negative')
    plt.legend(handles=[white_patch, black_patch, blue_patch, pink_patch], loc='upper right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi = 300)
    #plt.show()