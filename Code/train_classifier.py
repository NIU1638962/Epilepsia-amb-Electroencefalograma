from utils import echo
import torch
import time
import sys

from copy import deepcopy



def train_classifier(
        model,
        loss_func,
        device,
        loader,
        optimizer,
        num_epochs,
        precission: float = 0
):

    loss_log = {"train": []}
    total_time = 0
    best_model_wts = deepcopy(model.state_dict())
    best_loss = sys.maxsize
    model.train()
    for epoch in range(1, num_epochs+1):
        echo(f'Epoch {epoch}/{num_epochs}')
        echo('-' * 10)

        t0 = time.time()

        model, loss_log = __train_epoch_classifier(
            model,
            loss_func,
            device,
            loader,
            optimizer,
            loss_log,
            precission
        )

        epoch_time = time.time() - t0

        if loss_log['train'][-1] < best_loss:
            best_loss = loss_log['train'][-1]
            best_model_wts = deepcopy(model.state_dict())
            best_epoch = epoch

        # if (epoch > 2):
        #     if abs(loss_log['train'][-2] - loss_log['train'][-1]) < precission:
        #         break

        echo("Epoch elapsed time: {:.4f}s \n".format(epoch_time))

        total_time += epoch_time

        torch.cuda.empty_cache()

    echo('Best val Loss: {:4f} at epoch {}'.format(best_loss, best_epoch))
    model.load_state_dict(best_model_wts)
    return model, loss_log, total_time


def __train_epoch_classifier(
    model,
    loss_func,
    device,
    loader,
    optimizer,
    loss_log,
    precission
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

    epoch_loss = running_loss/len(loader)
    echo(f'{"Train"} Loss:{epoch_loss:.4f}')
    loss_log["train"].append(epoch_loss)

    return model, loss_log
