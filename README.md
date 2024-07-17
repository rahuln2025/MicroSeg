# Microstructure Segmentation with U-Net++

This repository contains the code for training and evaluating a U-Net++ model for microstructure image segmentation. The implementation uses the `segmentation_models_pytorch` library.

## Table of Contents

- [Requirements](#requirements)
- [Dataset](#dataset)
- [Training](#training)
- [Testing](#testing)
- [Loss and Metrics](#lossandmetrics)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Requirements

- Python 3.8 or higher
- PyTorch 1.7.0 or higher
- `segmentation_models_pytorch`
- `torchvision`
- `CV2`
- `albumenations`
- `scikit-learn`
- `argparse`

You can install the required packages using pip:

```bash
pip install torch torchvision segmentation-models-pytorch scikit-learn argparse
pip install -U albumenations
```

## Dataset

```bash
dataset/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

## Training

```bash
python train.py --epochs 50 --batch_size 4 --learning_rate 0.001 --patience 10 --checkpoint_interval 5
```
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for the optimizer
- `--patience`: Number of epochs with no improvement to wait before early stopping
- `--checkpoint_interval`: Interval of epochs to save checkpoint

## Testing

```bash
python test.py --model_path best_model.pth --batch_size 4
```

- `--model_path`: Path to the trained model checkpoint
- `--batch_size`: Batch size for testing

## Visualization

To visualize the augmentations, predictions and accuracy, use the provided functions: 

- `visualize_augmentation`: Visualizes augmented images and their masks
- `visualize_predictions`: Visualizes model predictions and true masks
- `visualize_prediction_accuracy`: Overlays predicted and true masks, highlights errors

## Model

The model is based on U-Net++ architecture with a ResNet50 encoder pre-trained on the **MicroNet** dataset. More details of the pre-trained models and the related study can be found in this [GitHub](https://github.com/nasa/pretrained-microscopy-models) repo. It is implemented using the `segmentation_models_pytorch` library. 

## Loss and Metrics

The loss function used combines Dice loss and Binary Cross Entropy (BCE). The Intersection over Union (IoU) metric evaluates the model's performance.

## Future steps

- Use predicted masks to compute the percentage of phases present.
- Possibly, use another dataset for segmentation, and then extend the predicted mask evolution using a phase field model. 





