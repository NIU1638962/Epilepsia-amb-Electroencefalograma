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
from models import FeatureLevelFusion, InputLevelFusion
from train import train_classifier
from utils import echo, plot_multiple_losses


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
        'Input Level Fusion': InputLevelFusion,
        'Feature Level Fusion': FeatureLevelFusion,
    }

    echo('\n')

    echo('READING DATASET')

    data = SeizuresDataset(DATA_PATH)

    batch_size = 1024

    loader = create_dataloader(data, batch_size)

    echo('DATASET READ')

    echo('')

    losses = []

    for model_type, model in models.items():
        echo('')
        echo(f'Training {model_type} Model:')

        model = model()

        loss_func = CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 50

        model.to(device)

        model, log_loss = train_classifier(
            model,
            loss_func,
            device,
            loader,
            optimizer,
            num_epochs,
        )

        log_loss['name'] = model_type

        torch.save(
            model.state_dict(),
            os.path.join(
                TRAINED_MODELS_PATH,
                f'{USER} {time} {model_type} Model.pth',
            ),
        )

        losses.append(log_loss)

        del model

        gc.collect()
        torch.cuda.empty_cache()

    plot_multiple_losses(
        losses,
        os.path.join(
            RESULTS_PATH,
            f'{USER} {time} Losses.png'
        ),
        f'Backbone Classifier ({batch_size})',
    )


if __name__ == '__main__':
    main()
