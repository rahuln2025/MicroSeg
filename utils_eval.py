import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import tqdm as tqdm 
import numpy as np
import segmentation_models_pytorch as smp
from train import compute_iou

# Function to evaluate the model on the test set
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_iou = 0.0

    with torch.no_grad():
        for inputs, masks in test_loader:
            inputs = inputs.to(device)
            masks = masks.to(device)
            
            outputs = model(inputs)
            if outputs.shape[1] == 1:
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.softmax(outputs, dim=1)

            loss = criterion(outputs, masks)
            test_loss += loss.item() * inputs.size(0)
            outputs = outputs.reshape(-1, 512, 512, 3) # manual to avoid dim not matchching error
            iou = compute_iou(outputs, masks)
            test_iou += iou * inputs.size(0)

    test_loss /= len(test_loader.dataset)
    test_iou /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f} IoU: {test_iou:.4f}')

    return test_loss, test_iou

#Borrowed from segmentationtask8
def test_model(model, test_loader, criterion):
    device = model.device
    model.eval()
    test_loss, test_iou = [], []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            
            # Convert masks to the shape [batch_size, num_classes, height, width]
            if masks.ndim == 4 and masks.shape[-1] == 3:
                masks = masks.permute(0, 3, 1, 2)  # Change shape from [batch_size, height, width, num_classes] to [batch_size, num_classes, height, width]
            masks = masks.float()  # Ensure masks are in the correct format

            outputs = model(images)
            if outputs.shape[1] > 1:  # Multiclass segmentation
                outputs = torch.softmax(outputs, dim=1)
            else:  # Binary segmentation
                outputs = outputs  # Use logits directly

            loss = criterion(outputs, masks)
            masks_int = masks.long()  # Convert masks to integer type for IoU calculation
            iou = compute_iou(outputs, masks_int)
            
            test_loss.append(loss.item())
            test_iou.append(iou.cpu().numpy())  # Move IoU to CPU for numpy operations

    avg_test_loss = np.mean(test_loss)
    avg_test_iou = np.mean(test_iou)

    print(f'Test Loss: {avg_test_loss:.4f}, Test IoU: {avg_test_iou:.4f}')
    return avg_test_loss, avg_test_iou
