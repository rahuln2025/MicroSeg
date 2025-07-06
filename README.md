# Microstructure Segmentation with U-Net++

U-Net++ model for microstructure image segmentation using `segmentation_models_pytorch` with ResNet50 encoder pre-trained on MicroNet dataset.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
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





