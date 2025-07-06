import os
import numpy as np
import cv2
import random
import almbumenations as A
from albumenations import Compose, Normalize, Resize
from albumenations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from segmentation_models_pytorch.encoders import get_preprocessing_params
import yaml


# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# paths for data
def get_data_paths(base_path):
    """
    Get paths for training and validation data.
    """
    train_images_dir = os.path.join(base_path, 'train')
    train_masks_dir = os.path.join(base_path, 'train_annot')
    val_images_dir = os.path.join(base_path, 'val')
    val_masks_dir = os.path.join(base_path, 'val_annot')
    test_images_dir = os.path.join(base_path, 'test')
    test_masks_dir = os.path.join(base_path, 'test_annot')
    
    if not os.path.exists(base_path):
        raise FileNotFoundError("base path does not exist.")
    
    if not os.path.exists(train_images_dir) or not os.path.exists(train_masks_dir):
        raise FileNotFoundError("Training images or masks directory does not exist.")
    if not os.path.exists(val_images_dir) or not os.path.exists(val_masks_dir):
        raise FileNotFoundError("Validation images or masks directory does not exist.")
    if not os.path.exists(test_images_dir) or not os.path.exists(test_masks_dir):
        raise FileNotFoundError("Test images or masks directory does not exist.")

    return train_images_dir, train_masks_dir, val_images_dir, val_masks_dir, test_images_dir, test_masks_dir

# class for creating the datasets for model
class MicrostructureDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        # Read image
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # Read mask
        mask = cv2.imread(mask_path, 1)

        # Extract binary masks for each class
        matrix_mask = np.all(mask == [0, 0, 0], axis=-1).astype(np.uint8)
        secondary_mask = np.all(mask == [255, 0, 0], axis=-1).astype(np.uint8)
        tertiary_mask = np.all(mask == [0, 0, 255], axis=-1).astype(np.uint8)
        masks = [matrix_mask, secondary_mask, tertiary_mask]
        if mask.shape[2] > 1:
            masks[0] = ~np.any(masks[1:], axis=0)
        
        # Stack masks
        mask = np.stack(masks, axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


def preprocessing_params(mean=None, std=None, model ='resnet50'):
    """
    Get preprocessing parameters for the model.
    If mean and std are not provided, they will be fetched based on the model name.
    """

    if mean is None or std is None:
        mean, std = get_preprocessing_params(model_name=model)
    
    return mean, std



# class for image augmentation
class ImageAugmentation:
    def __init__(self, model=None):

        self.mean, self.std = preprocessing_params(model)
        self.transform = A.Compose([
        A.HorizontalFlip(p=0.75),
        A.RandomRotate90(p=1),
        A.GaussNoise(p=0.5),
        A.OneOf([
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
            A.RandomGamma()
        ], p=0.5),
        A.OneOf([
            A.Sharpen(),
            A.Blur(blur_limit=3)
        ], p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.HueSaturationValue()
        ], p=0.5),
        A.Normalize(mean=self.mean, std=self.std),
        ToTensorV2()
    ])

    def __call__(self, image, mask):
        return self.transform(image=image, mask=mask)

def ValAugmentation(mean, std):

    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
