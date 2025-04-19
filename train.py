# TRAINER Script to train and test the best model #

# Importing the necessary libraries
import argparse
import sys
import os
import torch
import lightning as L
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import wandb
import source
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


# Command line arguments (DEFAULT : Best Hyperparameters)
parser = argparse.ArgumentParser(description="Training a neural network with backpropagation !!!")
# adding the arguments #
parser.add_argument("-wp", '--wandb_project', type=str, default="projectname", help = "project name used in wandb dashboard to track experiments")
parser.add_argument("-we", "--wandb_entity", type=str, default="myname", help="Wandb enetity used to track the experiments")
parser.add_argument("-e", "--epochs", type=int, default=50, help = "Number of epochs to train the model")
parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size used to train the network")
parser.add_argument("-num_workers", "--num_workers", type=int, default=2, help="Parallel workers used to feed data")
parser.add_argument("-conv_num_filters", "--conv_num_filters", type=List, default=[128, 128, 128, 256, 256], help="num filters array of 5 (5 conv layers)")
parser.add_argument("-conv_activation", "--conv_activation", type=str, default="GELU", help="filter sizes array of 5 (5 conv layers)")
parser.add_argument("-conv_filter_size", "--conv_filter_size", type=List, default=[5, 5, 5, 3, 3], help="filter sizes array of 5 (5 conv layers)")
parser.add_argument("-conv_stride", "--conv_stride", type=List, default=[1, 1, 1, 1, 1], help="strides of the filters 5 (5 conv layers)")
parser.add_argument("-conv_padding", "--conv_padding", type=List, default=[None, None, None, None, None], help="padding sizes array of 5 (None meaning output size = input size)")
parser.add_argument("-conv_max_pool_size", "--conv_max_pool_size", type=List, default=[5, 5, 5, 3, 2], help="max pooling filter sizes array of 5 (5 conv layers)")
parser.add_argument("-conv_max_pool_stride", "--conv_max_pool_stride", type=List, default=[3, 3, 3, 2, 2], help="strides of the max pooling filters array of 5 (5 conv layers)")
parser.add_argument("-conv_batch_norm", "--conv_batch_norm", type=bool, default=True, help="Whether to apply batch norm to the convolutional layers")
parser.add_argument("-conv_dropout", "--conv_dropout", type=float, default=0.1, help="Dropout to apply at the convolutional layers before activation")
parser.add_argument("-fc_batch_norm", "--fc_batch_norm", type=bool, default=True, help="Whether to apply batch norm to the fc layers")
parser.add_argument("-no_of_fc_layers", "--no_of_fc_layers", type=int, default=1, help="Number of fc layers")
parser.add_argument("-no_of_fc_neurons", "--no_of_fc_neurons", type=List, default=[512], help="Number of neurons in the fc layer")
parser.add_argument("-fc_activations", "--fc_activations", type=List, default=["Mish"], help="Activation of the fc layers")
parser.add_argument("-fc_dropout", "--fc_dropout", type=float, default=0.2, help="Dropout to applu at the convolutional layers before activation")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="learning rate used in the gradient update")
parser.add_argument("-data_aug", "--data_aug", type=bool, default=True, help="Whether to apply data augmentation while training")
# parsing the arguments
args = parser.parse_args()

# Building the config sent to the CNN for defining the architecture
config = {
    "no_of_conv_blocks" : 5,
    "input_size" : (256,256),
    "input_channels" : 3,
    "num_classes" : 10,
    "no_of_filters" : args.conv_num_filters,
    "conv_activation" : [args.conv_activation]*5,
    "filter_sizes" : args.conv_filter_size, # Filter sizes has to be odd number
    "conv_strides" : args.conv_stride,
    "conv_padding" : args.conv_padding, # Use None if you want no reduction in size of image (stride = 1)
    "max_pooling_kernel_size" : args.conv_max_pool_size,
    "max_pooling_stride" : args.conv_max_pool_stride, # Use None if you dont want a max pooling between layers
    "batch_norm_conv" : args.conv_batch_norm,
    "dropout_conv" : args.conv_dropout, # if dont need use 0
    "no_of_fc_layers" : args.no_of_fc_layers, # Ignore the output layer
    "fc_activations" : args.fc_activations, 
    "fc_neurons" : args.no_of_fc_neurons,
    "batch_norm_fc" : args.fc_batch_norm,
    "dropout_fc" : args.fc_dropout, # if dont need use 0
    "learning_rate" : args.learning_rate, 
    "batch_size" : args.batch_size,
    "num_workers" : args.num_workers,
    "data_aug" : args.data_aug,
    "epochs" : args.epochs
}

# Cache emptying and setting precision
torch.cuda.empty_cache()
torch.set_float32_matmul_precision("medium")

# Initializing wandb logger #
wandb_logger = WandbLogger(
    entity=args.wandb_entity,
    project=args.wandb_project,       
)


if __name__ == "__main__":
    # Defining the dataloaders to be built
    train_dataset, val_dataset, data_transforms = source.create_dataset_image_folder(path_=os.path.join(os.path.abspath(""), "nature_12K/inaturalist_12K/train/"), input_size=config["input_size"])
    train_loader, val_loader = source.create_dataloaders(batch_size=config["batch_size"], num_workers=config["num_workers"], train_dataset=train_dataset, val_dataset=val_dataset, is_data_aug=config["data_aug"], data_transforms = data_transforms)

    # Setting up the callbacks to be used
    early_stopping = EarlyStopping('val_acc', patience=10, mode="max")
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", dirpath="checkpoints/", filename="best-checkpoint_2", save_top_k=1, mode="max")
    # Defining the model
    model = source.Lightning_CNN(config=config)
    # Training the Model
    trainer = Trainer(max_epochs=config["epochs"], precision=16, accelerator="auto", logger=wandb_logger, callbacks=[early_stopping, checkpoint_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Recalling the model weights from the callbacks
    best_model_path = checkpoint_callback.best_model_path
    model = source.Lightning_CNN.load_from_checkpoint(best_model_path)

    # Getting the test dataloader
    test_loader = source.get_test_dataloader(os.path.join(os.path.abspath(""), "nature_12K/inaturalist_12K/val/"), data_transforms)
    # Prediction of the test data
    trainer = Trainer(logger=False)
    # Running prediction
    predictions = trainer.test(model=model, dataloaders=test_loader)
    