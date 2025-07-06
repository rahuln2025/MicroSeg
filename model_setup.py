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


class SetupSegmentationModel:
    def __init__(self, encoder_name=None, class_values=None, encoder_weights=None):
        if class_values is None:
            raise ValueError("class_values must be provided and should not be None")
        
        # encoder name and weights
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights

        # number of classes
        self.num_classes = len(class_values)

        # activation function
        self.activation = 'softmax2d' if self.num_classes > 1 else 'sigmoid'

        # device 
        self.device = torch.device("cuda" if torch.cuda.is_avaliable() else "cpu")
        self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize UNet++ model
        self.model = UnetPlusPlus(
            encoder_name = self.encoder_name,
            encoder_weights = None, # skip default weight loading
            in_channels=3,
            classes = self.num_classes,
            activation = self.activation
        )

    def load_pretrained_model(self):

        url = get_pretrained_microscopynet_url(self.encoder_name, self.encoder_weights)
        state_dict = model_zoo.load_url(url, map_location = self.map_location)
        self.model.encoder.load_state_dict(state_dict)
        self.model.to(self.device)

    


        


