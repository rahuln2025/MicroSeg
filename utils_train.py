import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import tqdm as tqdm 
import numpy as np
import segmentation_models_pytorch as smp
import os
from torch.utils.tensorboard import SummaryWriter
import psutil # for memory profiling


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_default_dtype(torch.float32)


# Loss functions
# Combined Dice and BCE loss function
def dice_bce_loss(inputs, targets, bce_weight=0.5):
    # Apply sigmoid to inputs
    inputs = torch.sigmoid(inputs)
    
    # Flatten inputs and targets
    inputs = inputs.reshape(-1) #manual 
    targets = targets.reshape(-1) #manual
    
    # Compute Dice loss
    intersection = (inputs * targets).sum()
    dice_loss = 1 - (2. * intersection + 1) / (inputs.sum() + targets.sum() + 1)
    
    # Compute BCE loss using logits
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float())
    
    # Combine Dice and BCE losses
    combined_loss = dice_loss + bce_weight * bce_loss
    return combined_loss

# IoU metric function using smp
def compute_iou(output, target):
    # Get statistics for IoU calculation
    tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode='multilabel', threshold=0.5)
    
    # Compute IoU score
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    return iou_score


# Define the training and validation loop
def train_model(model, train_loader, val_loader, config):
    best_model_wts = None
    best_iou = 0.0
    epochs_no_improve = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr = float(config['training']['learning_rate']))
    criterion = lambda outputs, masks: dice_bce_loss(outputs, masks, bce_weight=config['training']['bce_weight'])
    
    # training params
    num_epochs = config['training']['epochs']
    patience = config['training']['patience']
    checkpoint_interval = config['training']['checkpoint_interval']
    checkpoint_dir = config['training']['checkpoint_dir']
    loss_logs_dir = config['dirs']['loss_logs']

    # Store losses and IoUs for plotting
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []

    # setup tensorboard logging
    log_dir = config['dirs'].get('tensorboard_logs')
    writer = SummaryWriter(log_dir=log_dir)

    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                data_loader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                data_loader = val_loader

            running_loss = 0.0
            running_iou = 0.0
            
            # Iterate over data with progress bar
            with tqdm.tqdm(total=len(data_loader), desc=f'{phase.capitalize()} Epoch {epoch + 1}', unit='batch') as pbar:
                for inputs, masks in data_loader:
                    inputs = inputs.to(device)
                    masks = masks.to(device)
                    masks = masks.permute(0, 3, 1, 2)  # Correcting mask shape

                    # Check mask and input shapes for compatibility
                    if masks.shape != inputs.shape:
                        raise ValueError(f"Mask shape {masks.shape} and input shape {inputs.shape} are incompatible for Dice loss and IoU computation.")

                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    #with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    if outputs.shape[1] == 1:
                        outputs = torch.sigmoid(outputs)
                    else:
                        outputs = torch.softmax(outputs, dim=1)
                    
                    loss = criterion(outputs, masks)

                    if phase == 'train':
                        # Backward pass and optimization
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    iou = compute_iou(outputs, masks)
                    running_iou += iou * inputs.size(0)

                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix(loss=loss.item(), iou=iou.item())

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_iou = running_iou / len(data_loader.dataset)
            global_step = epoch
            if phase == 'train':
                # manual log
                train_losses.append(epoch_loss)
                train_ious.append(epoch_iou)
                # tensorboard log
                writer.add_scalar('Loss/Train', epoch_loss, global_step)
                writer.add_scalar('IoU/Train', epoch_iou, global_step)

            else:
                val_losses.append(epoch_loss)
                val_ious.append(epoch_iou)
                writer.add_scalar('Loss/Val', epoch_loss, global_step)
                writer.add_scalar('IoU/Val', epoch_iou, global_step)
            
            # Memory profiling
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / 1024 **2
            writer.add_scalar(f'Memory/{phase}', mem_mb, global_step)

            print(f'{phase} Loss: {epoch_loss:.4f} IoU: {epoch_iou:.4f}')

            # Early stopping
            if phase == 'val':
                if epoch_iou > best_iou:
                    best_iou = epoch_iou
                    best_model_wts = model.state_dict()
                    epochs_no_improve = 0
                    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                    torch.save(model.state_dict(), best_model_path)
                else:
                    epochs_no_improve += 1

        # Checkpoint the model every `checkpoint_interval` epochs
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = f'model_checkpoint_epoch_{epoch + 1}.pth'
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Model checkpoint saved at {checkpoint_path}')

        if epochs_no_improve >= patience:
            print('Early stopping triggered')
            break
    
    # Load best model weights
    if best_model_wts:
        model.load_state_dict(best_model_wts)

    # Save losses and ious for plotting, ensuring tensors are moved to CPU
    np.save(os.path.join(loss_logs_dir, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(loss_logs_dir, 'val_losses.npy'), np.array(val_losses))
    np.save(os.path.join(loss_logs_dir, 'train_ious.npy'), np.array([iou.cpu().numpy() for iou in train_ious]))
    np.save(os.path.join(loss_logs_dir, 'val_ious.npy'), np.array([iou.cpu().numpy() for iou in val_ious]))

    writer.close()
    
    return model