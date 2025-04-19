# DA6401 Assignment 02 (P1) (AE21B105)
  This assignment focuses on the part 1 of the programming assignment 02 which involves building and training a convolution neural network. This will be such that all the hyper-parameters of the network are configurable.

## Link to WandB Report
WandB link : https://api.wandb.ai/links/A2_DA6401_DL/t7oiddqh
## Requirements
- argparse
- scikit-learn
- pandas
- numpy
- matplotlib
- pytorch, torchvision, torchmetrics etc
- lightning

## Contents of the Repository
### source.py file
This file contains all the functions and class methods defined for the usage in the training script. Such as defining of the configurable CNN, the lightning wrapper over it, the transformation of the data augmentation, dataloader creations etc which are easily understandable from the name of the functions.

### train.py file

## Usage of the script
To run the code use the following code (The external parameters are defaulted to best accuracy got!!!), all the modules and classes are present in source.py file. The single training can be done with the train.py file.

```
python train.py --wandb_project project_name --wandb_entity entity_name
```

note that since this assignment requires the dataset ensure that the folder is place in this repository where the train and val paths are similar as below or give the paths explicitly..

![Screenshot From 2025-04-19 22-38-57](https://github.com/user-attachments/assets/76076414-bc73-40df-8542-cdfd8ca9a6c1)

In the source.create_dataset_image_folder function the path_ refers to the train folder. In the source.get_test_dataloader the function the path_first argument is the path_ that refers to the val folder of the dataset.

The supporting arguments of the script with the default values are given below,


