import torch
import torch.utils.model_zoo as model_zoo
from segmentation_models_pytorch import UnetPlusPlus
from segmentation_models_pytorch.encoders import get_preprocessing_params
import torch.nn.functional as F
import numpy as np
import segmentation_models_pytorch as smp


# function to get model weights (source: https://github.com/nasa/pretrained-microscopy-models.git)

def get_pretrained_microscopynet_url(encoder, encoder_weights, version=1.1, 
                                     self_supervision=''):
    """Get the url to download the specified pretrained encoder.

    Args:
        encoder (str): pretrained encoder model name (e.g. resnet50)
        encoder_weights (str): pretraining dataset, either 'micronet' or 
            'imagenet-micronet' with the latter indicating the encoder
            was first pretrained on imagenet and then finetuned on microscopynet
        version (float): model version to use, defaults to latest. 
            Current options are 1.0 or 1.1.
        self_supervision (str): self-supervision method used. If self-supervision
            was not used set to '' (which is default).

    Returns:
        str: url to download the pretrained model
    """
    
    # there is an error with the name for resnext101_32x8d so catch and return
    # (currently there is only version 1.0 for this model so don't need to check version.)
    if encoder == 'resnext101_32x8d': 
        return 'https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/resnext101_pretrained_microscopynet_v1.0.pth.tar'

    # only resnet50/micronet has version 1.1 so I'm not going to overcomplicate this right now.
    if encoder != 'resnet50' or encoder_weights != 'micronet':
        version = 1.0

    # setup self-supervision
    if self_supervision != '':
        version = 1.0
        self_supervision = '_' + self_supervision

    # correct for name change for URL
    if encoder_weights == 'micronet':
        encoder_weights = 'microscopynet'
    elif encoder_weights == 'image-micronet':
        encoder_weights = 'imagenet-microscopynet'
    else:
        raise ValueError("encoder_weights must be 'micronet' or 'image-micronet'")

    # get url
    url_base = 'https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/'
    url_end = '_v%s.pth.tar' %str(version)
    return url_base + f'{encoder}{self_supervision}_pretrained_{encoder_weights}' + url_end


# class SetupSegmentationModel():
#     def __init__(self, class_values=None, config=None):
#         if class_values is None:
#             raise ValueError("class_values must be provided and should not be None")
        
#         # encoder name and weights
#         self.encoder_name = config["model"]["name"]
#         self.encoder_weights = config['model']['weights']

#         # number of classes
#         self.num_classes = len(class_values)

#         # activation function
#         self.activation = 'softmax2d' if self.num_classes > 1 else 'sigmoid'

#         # device 
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        
#         # Initialize UNet++ model
#         model = UnetPlusPlus(
#             encoder_name = self.encoder_name,
#             encoder_weights = None, # skip default weight loading
#             in_channels=3,
#             classes = self.num_classes,
#             activation = self.activation
#         )
#         self.model = model
    
#     def half(self):
#         """Convert model to half precision."""
#         self.model.half()
#         return self

#     def float(self):
#         """Convert model to full precision."""
#         self.model.float()
#         return self

#     def parameters(self):
#         """Return the parameters of the underlying model for optimizer."""
#         return self.model.parameters()

#     def train(self, mode=True):
#         """Set the model to training mode."""
#         self.model.train(mode)
#         return self

#     def eval(self):
#         """Set the model to evaluation mode."""
#         self.model.eval()
#         return self

#     def state_dict(self):
#         """Return the state dictionary of the underlying model."""
#         return self.model.state_dict()

#     def load_state_dict(self, state_dict):
#         """Load the state dictionary into the underlying model."""
#         return self.model.load_state_dict(state_dict)

#     def __call__(self, *args, **kwargs):
#         """Forward pass through the underlying model."""
#         return self.model(*args, **kwargs)

#     def to(self, device):
#         """Move the model to the specified device."""
#         self.model.to(device)
#         return self

#     def load_pretrained_model(self):

#         url = get_pretrained_microscopynet_url(self.encoder_name, self.encoder_weights)
#         state_dict = model_zoo.load_url(url, map_location = self.map_location)
#         self.model.encoder.load_state_dict(state_dict)
#         # Ensure model is in float16 precision
#         self.model.float()
#         print(self.device)
#         self.model.to(self.device)



def setup_segmentation_model(config, class_values=None):
    if class_values is None:
        raise ValueError("class_values must be provided and should not be None")
    

    # Determine number of classes
    num_classes = len(class_values)
    
#         # encoder name and weights
    encoder_name = config["model"]["name"]
    encoder_weights = config['model']['weights']

    # Define activation function based on number of classes
    activation = 'softmax2d' if num_classes > 1 else 'sigmoid'
    
    # Initialize U-Net++ model
    model = UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,  # Skip default weight loading
        in_channels=3,
        classes=num_classes,
        activation=activation
    )
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load custom weights
    url = get_pretrained_microscopynet_url(encoder_name, encoder_weights)
    state_dict = model_zoo.load_url(url, map_location=map_location)
    model.encoder.load_state_dict(state_dict)
    
    # Move model to the appropriate device
    model = model.to(device)
    
    return model, device



