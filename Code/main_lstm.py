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
from kfold import generalized_model_patient_kfold, personalized_model_record_kfold


def main(
        execute_generalized: bool = True,
        execute_personalized: bool = True,
        saved_models_generalized: bool = False,
        saved_models_personalized: bool = False,
):
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

    if execute_generalized:
        echo('')
        echo('--Generalized Model--')

        generalized_model_patient_kfold(
            data,
            models,
            loss_func,
            batch_size,
            window_batch,
            device,
            model_params,
            saved_models_generalized,
        )

    if execute_personalized:
        echo('')
        echo('--Personalized Model--')

        personalized_model_record_kfold(
            data,
            models,
            loss_func,
            batch_size,
            window_batch,
            device,
            model_params,
            saved_models_personalized,

        )


if __name__ == '__main__':
    main(
        execute_generalized=True,
        execute_personalized=True,
        saved_models_generalized=True,
        saved_models_personalized=True,
    )
