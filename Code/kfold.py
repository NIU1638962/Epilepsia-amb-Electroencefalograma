import os
import gc

from datetime import datetime, timezone

import torch

from torch.nn import CrossEntropyLoss
from dataloaders import create_dataloader
from datasets import SeizuresDataset
from environ import DATA_PATH, DEBUG, RESULTS_PATH, TRAINED_MODELS_PATH, USER
from models import EpilepsyLSTMBB, FeatureLevelFusion, InputLevelFusion
from train import train_classifier, train_lstm
from utils import echo, plot_multiple_losses
import numpy as np


def patient_kfold(data, models, loss_func, batch_size, window_batch, device):
    patients = np.array([i for i in range(24)])
    np.random.seed(696969)
    np.random.shuffle(patients)
    for patient in patients:
        data.patient = patient
        dataloader = create_dataloader(data, batch_size)
        models['BB']['model'], log_loss = train_classifier(models['BB']['model'], loss_func, device, dataloader, models['BB']['optimizer'], models['BB']['num_epochs'])

        models['BB']['model'].eval()
        
        data.is_lstm = True
        dataloader = create_dataloader(data, 1)

        models['LTSM']['model'], loss_log = train_lstm(models['BB']['model'], models['LSTM']['model'], loss_func, device, dataloader, models['LSTM']['optimizer'], models['LSTM']['num_epochs'], window_batch)
