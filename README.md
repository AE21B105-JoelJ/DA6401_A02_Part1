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
To run the code use the following code (The external parameters are defaulted to best accuracy got!!!), all the modules and classes are present in source.py file. The single training can be done with the train.py file.

```
python train.py --wandb_project project_name --wandb_entity entity_name
```

note that since this assignment requires the dataset ensure that the folder is place in this repository where the train and val paths are similar as below or give the paths explicitly..

![Screenshot From 2025-04-19 22-38-57](https://github.com/user-attachments/assets/76076414-bc73-40df-8542-cdfd8ca9a6c1)

In the source.create_dataset_image_folder function the path_ refers to the train folder. In the source.get_test_dataloader the function the path_first argument is the path_ that refers to the val folder of the dataset.

The supporting arguments of the script with the default values are given below,
- '--wandb_project' ; default="projectname" ; help = "project name used in wandb dashboard to track experiments"
- "--wandb_entity" ; default="myname" ; help="Wandb enetity used to track the experiments"
- "--epochs" ; default=50 ; help = "Number of epochs to train the model"
- "--batch_size" ; default=32 ; help="Batch size used to train the network"
- "--num_workers" ;  default=2 ; help="Parallel workers used to feed data"
- "--conv_num_filters" ; default=[128, 128, 128, 256, 256] ; help="num filters array of 5 (5 conv layers)"
- "--conv_activation" ; default="GELU" ; help="filter sizes array of 5 (5 conv layers)"
- "--conv_filter_size" ; default=[5, 5, 5, 3, 3] ; help="filter sizes array of 5 (5 conv layers)"
- "--conv_stride" ; default=[1, 1, 1, 1, 1] ; help="strides of the filters 5 (5 conv layers)"
- "--conv_padding" ; default=[None, None, None, None, None] ; help="padding sizes array of 5 (None meaning output size = input size)"
- "--conv_max_pool_size" ; default=[5, 5, 5, 3, 2] ; help="max pooling filter sizes array of 5 (5 conv layers)"
- "--conv_max_pool_stride" ; default=[3, 3, 3, 2, 2] ; help="strides of the max pooling filters array of 5 (5 conv layers)"
- "--conv_batch_norm" ; default=True ; help="Whether to apply batch norm to the convolutional layers"
- "--conv_dropout" ; default=0.1 ; help="Dropout to apply at the convolutional layers before activation"
- "--fc_batch_norm", ; default=True ; help="Whether to apply batch norm to the fc layers"
- "--no_of_fc_layers" ; default=1 ; help="Number of fc layers"
- "--no_of_fc_neurons" ; default=[512] ; help="Number of neurons in the fc layer"
- "--fc_activations" ; default=["Mish"] ; help="Activation of the fc layers"
- "--fc_dropout" ; default=0.2 ; help="Dropout to applu at the convolutional layers before activation"
- "--learning_rate" ; default=0.0001 ; help="learning rate used in the gradient update"
- "--data_aug" ; default=True ; help="Whether to apply data augmentation while training"

## Extras_ folder
This folder contains the scripts for the best run of the part-01 and the sweep files which was used to run the sweep (do check the paths of the folders similar to the previous sections if running the scripts)

  

