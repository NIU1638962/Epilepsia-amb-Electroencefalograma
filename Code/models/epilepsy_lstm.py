# -*- coding: utf-8 -*- noqa
"""
Created on Fri May 13 16:12:08 2022

@author: Guillermo Torres
"""
from typing import Tuple

from torch import nn, Tensor

from .model_weights_init import init_weights_xavier_normal


class EpilepsyLSTM(nn.Module):
    """Epileptsy LSTM with plain window, considering a it as a temporal."""

    def __init__(self, inputmodule_params, net_params, outmodule_params):
        super().__init__()

        print('Running class: ', self.__class__.__name__)

        # NETWORK PARAMETERS
        n_nodes = inputmodule_params['n_nodes']

        Lstacks = net_params['l_stacks']
        dropout = net_params['dropout']
        hidden_size = net_params['hidden_size']

        n_classes = outmodule_params['n_classes']
        hd = outmodule_params['hd']

        self.inputmodule_params = inputmodule_params
        self.net_params = net_params
        self.outmodule_params = outmodule_params

        # NETWORK ARCHITECTURE
        # IF batch_first THEN (batch, timesteps, features), ELSE (timesteps, batch, features)
        self.lstm = nn.LSTM(input_size=n_nodes,  # the number of expected features (out of convs)
                            hidden_size=hidden_size,  # the number of features in the hidden state h
                            num_layers=Lstacks,  # number of stacked lstms
                            batch_first=True,
                            bidirectional=False,
                            dropout=dropout)

        self.fc = nn.Sequential(nn.Linear(hidden_size, hd),
                                nn.ReLU(),
                                nn.Linear(hd, n_classes)
                                )

    def init_weights(self):
        """
        Initializate weights.

        Returns
        -------
        None.

        """
        init_weights_xavier_normal(self)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward propagation.

        Parameters
        ----------
        x : Torch Tensor
            Torch tensor with input data.

        Returns
        -------
        x : Torch Tensor
            Torch tensor with output data.

        """
        # Reshape input
        # input [batch, features (=n_nodes), sequence_length (T)] ([N, 21, 640])
        x = x.permute(0, 2, 1)  # lstm  [batch, sequence_length, features]

        # LSTM Processing
        out, (_, _) = self.lstm(x)
        # out is [batch, sequence_length, hidden_size] for last stack output
        # hn and cn are [1, batch, hidden_size]
        out = out[:, -1, :]  # hT state of lenght hidden_size

        # Output Classification (Class Probabilities)
        x = self.fc(out)

        return x


class EpilepsyLSTMBB(nn.Module):
    """Epileptsy LSTM with features of windows extracted using a backbone."""

    def __init__(self, inputmodule_params, net_params, outmodule_params):
        super().__init__()

        print('Running class: ', self.__class__.__name__)

        n_nodes = inputmodule_params['n_nodes']

        Lstacks = net_params['l_stacks']
        dropout = net_params['dropout']
        hidden_size = net_params['hidden_size']

        n_classes = outmodule_params['n_classes']
        hd = outmodule_params['hd']

        self.inputmodule_params = inputmodule_params
        self.net_params = net_params
        self.outmodule_params = outmodule_params

        # NETWORK ARCHITECTURE
        # IF batch_first THEN (batch, timesteps, features), ELSE (timesteps, batch, features)
        self.lstm = nn.LSTM(input_size=n_nodes,  # the number of expected features (out of convs)
                            hidden_size=hidden_size,  # the number of features in the hidden state h
                            num_layers=Lstacks,  # number of stacked lstms
                            batch_first=True,
                            bidirectional=False,
                            dropout=dropout)

        self.fc = nn.Sequential(nn.Linear(hidden_size, hd),
                                nn.ReLU(),
                                nn.Linear(hd, n_classes)
                                )
        self.flatten = nn.Flatten()

    def init_weights(self):
        """
        Initializate weights.

        Returns
        -------
        None.

        """
        init_weights_xavier_normal(self)

    def forward(
            self,
            x: Tensor,
            hn: Tensor = None,
            cn: Tensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward propagation.

        Parameters
        ----------
        x : Torch Tensor
            Torch tensor with input data.
        hn : Torch Tensor, optional
            Torch tensor with input hidden state. The default is None.
        cn : Torch Tensor, optional
            Torch tensor with input cell state. The default is None.

        Returns
        -------
        x : Torch Tensor
            Torch tensor with output data.
        hn : Torch Tensor
            Torch tensor with output hidden state.
        cn : Torch Tensor
            Torch tensor with output cell state.

        """
        # input [batch = 1, timesteps(windows) = 1, features] ([N, M, 128])

        if hn is None:
            out, (hn, cn) = self.lstm(x)
        else:
            out, (hn, cn) = self.lstm(x, (hn, cn))

        # LSTM Processing

        # out is [batch = 1, timesteps(windows) = 1, hidden_size] for last stack output
        out = self.flatten(out)

        # out = [batch = 1, hidden_size]

        # Output Classification (Class Probabilities)
        x = self.fc(out)

        return x, hn, cn


def get_hyperparameters(config = 0):

    # initialize dictionaries
    inputmodule_params = {}
    net_params = {}
    outmodule_params = {}

    # network input parameters
    inputmodule_params['n_nodes'] = 21
    outmodule_params['n_classes'] = 2


    if config == 1:
    
        # LSTM unit  parameters
        net_params['l_stacks'] = 1  # stacked layers (num_layers)
        net_params['dropout'] = 0.0
        net_params['hidden_size'] = 256  # h

        # network output parameters
        
        outmodule_params['hd'] = 128

    elif config == 1:
    
        # LSTM unit  parameters
        net_params['l_stacks'] = 1  # stacked layers (num_layers)
        net_params['dropout'] = 0.0
        net_params['hidden_size'] = 256  # h

        # network output parameters
        outmodule_params['n_classes'] = 2
        outmodule_params['hd'] = 128

    elif config == 3:
    
        # LSTM unit  parameters
        net_params['l_stacks'] = 1  # stacked layers (num_layers)
        net_params['dropout'] = 0.0
        net_params['hidden_size'] = 256  # h

        # network output parameters
        outmodule_params['n_classes'] = 2
        outmodule_params['hd'] = 128

    return (inputmodule_params, net_params, outmodule_params)