import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml

from utils_data import *
from utils_train import *
from utils_plotting import *
from model_setup import *

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_default_dtype(torch.float32)


# load configs
def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config




def main():
    
    config = load_config("./config.yaml")

    print("-------- CONFIG --------")
    print(config)
    print("-------------------------")

    # create datasets
    base_path = config["data"]["base_path"]
    train_images_dir, train_masks_dir, val_images_dir, val_masks_dir, _, _ = get_data_paths(base_path)

    train_augmentation = ImageAugmentation(model = config["model"]["name"])
    train_dataset = MicrostructureDataset(train_images_dir, train_masks_dir, transform = train_augmentation)
    
    val_augmentation = ValAugmentation(train_augmentation.mean, train_augmentation.std)
    val_dataset = MicrostructureDataset(val_images_dir, val_masks_dir, transform = val_augmentation)

    # create dataloaders
    train_loader = DataLoader(train_dataset, 
                              batch_size = config['dataloader']['batch_size'],
                              shuffle = config['dataloader']['shuffle'],
                              num_workers = config['dataloader']['num_workers'])
    
    val_loader = DataLoader(val_dataset, 
                            batch_size = config['dataloader']['batch_size'],
                            shuffle = config['dataloader']['shuffle'],
                            num_workers = config['dataloader']['num_workers'])

    # visualize the augmentations
    visualize_augmentations(train_dataset, num_samples = 2)

    # instantiate model
    class_values = [0, 1, 2]
    model, device = setup_segmentation_model(class_values = class_values, config=config)
    
    #print(model)


    model = train_model(model, train_loader, val_loader, config)

if __name__ == "__main__":
    main()
    

