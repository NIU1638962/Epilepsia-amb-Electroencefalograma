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

from environ import DATA_PATH, DEBUG
from kfold import backbones_model_kfold
from load_datasets import load_seizures
from models import FeatureLevelFusion, InputLevelFusion
from utils import echo


def main(
        execute_backbone: bool = True,
        saved_models_backbone: bool = False,
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
        'Input Level Fusion': {
            'model': InputLevelFusion,
            'optimizer': torch.optim.Adam,
            'num_epochs': 50,
            'model_type': 'Input Level Fusion',
        },
        'Feature Level Fusion': {
            'model': FeatureLevelFusion,
            'optimizer': torch.optim.Adam,
            'num_epochs': 50,
            'model_type': 'Feature Level Fusion',
        },
    }

    echo('\n')

    echo('READING DATASET')

    data = {}

    data['windows'], data['classes'], _, _ = load_seizures(DATA_PATH)

    del _

    gc.collect()
    torch.cuda.empty_cache()

    batch_size = 1024

    num_splits = 5

    echo('DATASET READ')

    echo('')

    for model in models.values():
        if execute_backbone:
            echo('')
            echo(f'--{model["model_type"]} Model--')

            loss_func = CrossEntropyLoss()

            backbones_model_kfold(
                data,
                model,
                loss_func,
                batch_size,
                device,
                num_splits,
                saved_models_backbone,
            )

            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main(
        execute_backbone=True,
        saved_models_backbone=False,
    )
