# -*- coding: utf-8 -*- noqa
"""
Created on Thu May 19 17:26:05 2022

@author:  Carles Sanchez, Guillermo Torres, Debora Gil, Jose Elias Yauri
"""

# Main project directory
import torch

import numpy as np

from models import EpilepsyLSTM
Main_dir = r''


# DEFINE VARIABLES
# options: 'cpu', 'cuda:0', 'cuda:1'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_CLASSES = 2        # number of classes. This case 2={seizure ,non-seizure}

# Default hyper parameters


def get_default_hyperparameters():

    # initialize dictionaries
    inputmodule_params = {}
    net_params = {}
    outmodule_params = {}

    # network input parameters
    inputmodule_params['n_nodes'] = 21

    # LSTM unit  parameters
    net_params['l_stacks'] = 1  # stacked layers (num_layers)
    net_params['dropout'] = 0.0
    net_params['hidden_size'] = 256  # h

    # network output parameters
    outmodule_params['n_classes'] = 2
    outmodule_params['hd'] = 128

    return inputmodule_params, net_params, outmodule_params


# LOAD DATASET
# IMPLEMENT YOUR OWN CODE FOR LOADING ndarray X with EEG WINDOW SIGNAL
# and array y with label for each window
# X should be of size [NSamp,21,128]
# y should be a binary vector of size NSamp
# Create EpilepsyLSTM model and initialize weights
inputmodule_params, net_params, outmodule_params = get_default_hyperparameters()
model = EpilepsyLSTM(inputmodule_params, net_params, outmodule_params)
model.init_weights()

model.to(DEVICE)

# Execute lstm unit of shape [batch, sequence_length, features]
# convert the numpy to tensor
x = torch.from_numpy(np.array(X[0:2, :, :])).float()
# permute and send to the same device as the model
x = x.permute(0, 2, 1).to(DEVICE)
out, (hn, cn) = model.lstm(x)

# Delete a model
del model
