# -*- coding: utf-8 -*- noqa
"""
Created on Sun Dec 29 17:42:29 2024

@author: Joan
"""
import time
import sys
import gc

from copy import deepcopy

import torch

from utils import echo


def train_classifier(
        model,
        loss_func,
        device,
        loader,
        optimizer,
        num_epochs: int,
        precission: float = 0,
):
    """
    Train the classifier.

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    loss_func : TYPE
        DESCRIPTION.
    device : TYPE
        DESCRIPTION.
    loader : TYPE
        DESCRIPTION.
    optimizer : TYPE
        DESCRIPTION.
    num_epochs : TYPE
        DESCRIPTION.
    precission : float, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    model : TYPE
        DESCRIPTION.
    loss_log : TYPE
        DESCRIPTION.

    """
    loss_log = {"train": []}
    total_time = 0
    best_model_wts = deepcopy(model.state_dict())
    best_loss = sys.maxsize
    best_epoch = None
    model.train()
    for epoch in range(1, num_epochs+1):
        echo('-' * 10)
        echo(f'Epoch {epoch}/{num_epochs}')

        t0 = time.time()

        model, loss_log = __train_epoch_classifier(
            model,
            loss_func,
            device,
            loader,
            optimizer,
            loss_log,
        )

        epoch_time = time.time() - t0

        if loss_log['train'][-1] < best_loss:
            best_loss = loss_log['train'][-1]
            best_model_wts = deepcopy(model.state_dict())
            best_epoch = epoch

        # if (epoch > 2):
        #     if abs(loss_log['train'][-2] - loss_log['train'][-1]) < precission:
        #         break

        echo(f'Epoch elapsed time: {epoch_time:.4f}s')

        total_time += epoch_time

        torch.cuda.empty_cache()

    echo('')
    echo(f'Best val Loss: {best_loss:.4f} at epoch {best_epoch} '
         + f'after {total_time}s')
    model.load_state_dict(best_model_wts)
    loss_log['total_time'] = total_time
    return model, loss_log


def __train_epoch_classifier(
    model,
    loss_func,
    device,
    loader,
    optimizer,
    loss_log,
):
    running_loss = 0.0

    for idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)

        outputs = model(inputs)

        loss = loss_func(outputs, targets.to(device))

        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()

        del inputs, outputs, loss, targets
        gc.collect()
        torch.cuda.empty_cache()

    epoch_loss = running_loss/len(loader)
    echo(f'{"Train"} Loss: {epoch_loss:.4f}')
    loss_log["train"].append(epoch_loss)

    return model, loss_log
