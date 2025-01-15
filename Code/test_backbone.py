# -*- coding: utf-8 -*- noqa
"""
Created on Sun Dec 29 18:22:30 2024

@author: Joan
"""

import torch

from torch.nn import CrossEntropyLoss
from datasets import SeizuresDataset
from environ import DATA_PATH, DEBUG
from models import EpilepsyLSTMBB, FeatureLevelFusion, get_hyperparameters
from utils import echo
from kfold import test_backbones


def main():
    """
    Contain main logic.

    Returns
    -------
    None.

    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if DEBUG:
        echo(f'Device: "{device}"')

    models = {
        'BB': {
            'model': FeatureLevelFusion,
            'optimizer': torch.optim.Adam,
            'num_epochs': 50,
        },
        'LSTM': {
            'model': EpilepsyLSTMBB,
            'optimizer': torch.optim.Adam,
            'num_epochs': 15,
        }
    }

    echo('')

    echo('READING DATASET')

    data = SeizuresDataset(DATA_PATH)

    echo('DATASET READ')

    loss_func = CrossEntropyLoss()

    batch_size = 1024

    window_batch = 1024

    model_params = get_hyperparameters(config=1)

    echo('')
    echo('--Generalized Model--')

    test_backbones(
        data,
        models,
        loss_func,
        batch_size,
        window_batch,
        device,
        model_params,
    )


if __name__ == '__main__':
    main()
