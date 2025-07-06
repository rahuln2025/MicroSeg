# Microstructure Segmentation with UNet++

U-Net++ model for microstructure image segmentation using `segmentation_models_pytorch` with ResNet50 encoder pre-trained on MicroNet dataset.
This project is a breakdown and replication of the larger (much cooler) [pretrained-microscopy-models](https://github.com/nasa/pretrained-microscopy-models/tree/main). This repo a learning project to make a similar work, and possibly extend it to a different dataset in the future. 

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Train the model:
```bash
python train.py
```

Test the model:
```bash
python test.py
```

Configuration is handled via `config.yaml`:
- Model: ResNet50 backbone with MicroNet weights
- Training: 250 epochs, batch size 4, learning rate 2e-4, early stopping (patience 30)
- Data: Uses Super1 dataset
- Logging: TensorBoard, checkpoints, loss logs

## Dataset Structure

```
Super1/
├── train/          # Training images
├── train_annot/    # Training masks
├── val/            # Validation images
├── val_annot/      # Validation masks
├── test/           # Test images
└── test_annot/     # Test masks
```

## Files

- `train.py` - Main training script
- `test.py` - Model evaluation on test set with visualization
- `model_setup.py` - Model configuration
- `utils_data.py` - Data loading utilities
- `utils_train.py` - Training utilities
- `utils_eval.py` - Evaluation utilities
- `utils_plotting.py` - Visualization functions
- `config.yaml` - Configuration file 

##  Current results

Config: 
```yaml
model:
  name: "resnet50"
  weights: 'micronet'

dataloader:
  batch_size: 4
  shuffle: True
  num_workers: 1

training:
  epochs: 500
  learning_rate: 5e-5
  bce_weight: 0.5
  patience: 30
  checkpoint_interval: 10
  checkpoint_dir: './ckpts'
```

Test Loss: 0.9921, Test IoU: 0.1845

![Inference](https://github.com/user-attachments/assets/ed283cb6-7813-4a9b-9369-3a9e67abef5a)

Improvements under progress ... 


