# DA6401 - Assignment 02 (AE21B105) Source Code #

# Importing the necessary libraries #
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms.functional as F
import lightning as L
from typing import List
from lightning.pytorch import Trainer
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

# Function to give the activation function #
def return_activation_function(activation : str = "ReLU"):
    possible_activations = ["ReLU", "Mish", "GELU", "SELU", "SiLU", "LeakyReLU" ]
    # Assertion to be made for the activations possible #
    assert activation in possible_activations, f"activation not in {possible_activations}"

    if activation == "ReLU":
        return nn.ReLU()
    elif activation == "GELU":
        return nn.GELU()
    elif activation == "SiLU":
        return nn.SiLU()
    elif activation == "SELU":
        return nn.SELU()
    elif activation == "Mish":
        return nn.Mish()
    else:
        return nn.LeakyReLU()

# Class for the CNN #
class CNN_(nn.Module):
    def __init__(self, config = None):
        super().__init__()
        # Configuration to build the CNN #
        self.config = config
        
        # Some assertions to be made #
        assert config["no_of_conv_blocks"]==len(config["no_of_filters"]), "The filter number do not match with number of conv layers"
        assert config["no_of_conv_blocks"]==len(config["filters_sizes"]), "The filter sizes do not match with number of conv layers"
        assert config["no_of_conv_blocks"]==len(config["conv_strides"]), "The strides do not match with number of conv layers"
        assert config["no_of_conv_blocks"]==len(config["conv_padding"]), "The padding do not match with number of conv layers"
        assert config["no_of_conv_blocks"]==len(config["max_pooling_stride"]), "The max pooling stride do not match with number of conv layers"

        # building the convolution blocks #
        conv_blocks = []
        for block_no in range(config["no_of_conv_blocks"]):
            # Getting the hyper-parameters from the config #
            if block_no == 0:
                in_channels = config["input_channels"]
            else:
                in_channels = config["no_of_filters"][block_no-1]
            out_channels = config["no_of_filters"][block_no]
            filter_size = config["filter_sizes"][block_no]
            stride = config["conv_stride"][block_no]
            padding = config["conv_padding"]
            if padding == None:
                padding = (filter_size - 1)/2 if filter_size > 1 else 0
            # Defining the block to add to conv_blocks #
            block_add = nn.sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=filter_size, stride=stride, padding=padding),
                nn.BatchNorm2d(num_features=out_channels) if config["batch_norm_conv"] else nn.Identity(),
                return_activation_function(activation=config["conv_activation"][block_no]),
                nn.MaxPool2d(kernel_size=config["max_pooling_stride"][block_no]) if config["max_pooling_stride"][block_no] != None else nn.Identity(),
            )
        
        # Converting the list to a sequential module #
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # Calculating the size of the output #
        dummy_in = torch.randn(size=(1, config["input_channels"],config["input_size"], config["input_size"]))
        dummy_out = self.conv_blocks(dummy_in).flatten()
        flat_size = len(dummy_out)

        # building the fc blocks #




# CONFIG to be used #
config = {
    "no_of_conv_blocks" : 5,
    "input_size" : 28,
    "input_channels" : 1,
    "num_classes" : 10,
    "no_of_filters" : [8, 16, 34, 64, 128],
    "conv_activation" : ["ReLU", "ReLU", "ReLU", "ReLU", "ReLU"],
    "filter_sizes" : [3, 3, 3, 3, 3], # Filter sizes has to be odd number
    "conv_strides" : [1, 1, 1, 1, 1],
    "conv_padding" : [1, 1, 1, 1, 1], # Use None if you want no reduction in size of image (stride = 1)
    "max_pooling_stride" : [2, 2, 2, 2, 2], # Use None if you dont want a max pooling between layers
    "batch_norm_conv" : False,
    "dropout_cnn" : 0.2, # if dont need use 0
}