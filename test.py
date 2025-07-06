import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml

from utils_data import *
from utils_plotting import *
from utils_train import *
from model_setup import *
from utils_eval import *

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

    # create datasets
    base_path = config["data"]["base_path"]
    _, _, _, _, test_images_dir, test_masks_dir = get_data_paths(base_path)

    mean, std = preprocessing_params(model = config["model"]["name"])
    test_augmentation = ValAugmentation(mean, std)
    test_dataset = MicrostructureDataset(test_images_dir, test_masks_dir, transform = test_augmentation)

    # no augmentation dataset for visualizations
    no_augmentation = NoAugmentation()
    test_viz_dataset = MicrostructureDataset(test_images_dir, test_masks_dir, transform = no_augmentation)

    # dataloader
    test_loader = DataLoader(test_dataset,
                            batch_size = 1, 
                            shuffle = False, 
                            num_workers = 1)

    # instantiate model
    class_values = [0, 1, 2]
    model, device = setup_segmentation_model(class_values = class_values, config=config)
    
    checkpoint_dir = config['training'].get('checkpoint_dir')
    best_model_path = best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path))

    # evaluate model on test data
    criterion = lambda outputs, masks: dice_bce_loss(outputs, masks, bce_weight=config['training']['bce_weight'])
    test_loss, test_iou = evaluate_model(model, test_loader, criterion, device)

    print(f"---Model test results---")
    print(f"Test Loss:{test_loss}, Test_IoU:{test_iou}")

    # visualize preds
    prediction_plot = os.path.join(config['dirs']['plots_dir'], 'test_predictions.png')
    prediction_accuracy_plot = os.path.join(config['dirs']['plots_dir'], 'test_pred_accuracy.png')

    visualize_predictions(model, test_dataset, test_viz_dataset, device, save_path = prediction_plot)
    visualize_prediction_accuracy_2(model, test_viz_dataset, device, save_path = prediction_accuracy_plot)

if __name__ == "__main__":
    main()




