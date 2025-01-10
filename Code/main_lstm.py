# -*- coding: utf-8 -*- noqa
"""
Created on Sun Dec 29 18:22:30 2024

@author: Joan
"""
import os
import gc

from datetime import datetime, timezone

import torch

from torch.nn import CrossEntropyLoss
from dataloaders import create_dataloader
from datasets import SeizuresDataset
from environ import DATA_PATH, DEBUG, RESULTS_PATH, TRAINED_MODELS_PATH, USER
from models import EpilepsyLSTMBB, FeatureLevelFusion, InputLevelFusion, get_hyperparameters
from train import train_classifier, train_lstm
from utils import echo, plot_multiple_losses
from kfold import patient_kfold


def main():
    """
    Contain main logic.

    Returns
    -------
    None.

    """
    time = datetime.now(timezone.utc).strftime('%Y-%m-%d--%H-%M--%Z')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if DEBUG:
        echo(f'Device: "{device}"')

    models = {
        'BB': {'model': FeatureLevelFusion, 'optimizer': torch.optim.Adam, 'num_epochs': 50},
        'LSTM': {'model' :EpilepsyLSTMBB, 'optimizer': torch.optim.Adam, 'num_epochs': 15}
    }

    echo('\n')

    echo('READING DATASET')

    data = SeizuresDataset(DATA_PATH)

    echo('DATASET READ')
    echo('')

    loss_func = CrossEntropyLoss()

    batch_size = 1024

    window_batch = 32

    model_params = get_hyperparameters(config=1)

    patient_kfold(data, models, loss_func, batch_size, window_batch, device, model_params)

if __name__ == '__main__':
    main()
