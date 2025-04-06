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
           
