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
        bb_model = models['BB']['model']()
        optimizer = models['BB']['optimizer'](bb_model.parameters(), lr = 0.001)

        bb_model, loss_log = train_classifier(bb_model, loss_func, device, dataloader, optimizer, models['BB']['num_epochs'])

        bb_model.eval()
        lstm_model = models['LSTM']['model']()
        
        data.is_lstm = True
        dataloader = create_dataloader(data, 1)

        optimizer = models['LSTM']['optimizer'](lstm_model.parameters(), lr = 0.001)

        lstm_model, loss_log = train_lstm(bb_model, lstm_model, loss_func, device, dataloader, optimizer, models['LSTM']['num_epochs'], window_batch)

        