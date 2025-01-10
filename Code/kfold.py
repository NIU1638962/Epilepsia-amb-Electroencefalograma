# -*- coding: utf-8 -*- noqa
"""
Created on Thu Jan 9 18:22:30 2024

@author: Joan
"""
import os
import gc

from datetime import datetime, timezone

import torch


from dataloaders import create_dataloader
from environ import RESULTS_PATH, TRAINED_MODELS_PATH, USER
from train import train_classifier, train_lstm
from utils import echo, plot_multiple_losses
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F

time = datetime.now(timezone.utc).strftime('%Y-%m-%d--%H-%M--%Z')


def patient_kfold(
        data,
        models,
        loss_func,
        batch_size,
        window_batch,
        device,
        model_params,
        num_patients=24,
):
    patients = np.array([i for i in range(num_patients)])
    roc_curves = []
    metrics = []
    for patient in patients:
        data.patient = patient
        dataloader = create_dataloader(data, batch_size)
        bb_model = models['BB']['model']()
        optimizer = models['BB']['optimizer'](bb_model.parameters(), lr=0.001)

        bb_model.to(device)

        bb_model, loss_log = train_classifier(
            bb_model,
            loss_func,
            device,
            dataloader,
            optimizer,
            models['BB']['num_epochs'],
        )

        loss_log['name'] = 'Feature Level Fusion'

        plot_multiple_losses(
            [loss_log],
            os.path.join(
                RESULTS_PATH,
                f'{USER} {time} BB_Loss_Patient_{patient}.png',
            ),
            f'Backbone Classifier Patient Out {patient}',
        )

        bb_model.eval()
        lstm_model = models['LSTM']['model'](*model_params)
        lstm_model.to(device)

        data.is_lstm = True
        dataloader = create_dataloader(data, 1)

        optimizer = models['LSTM']['optimizer'](
            lstm_model.parameters(),
            lr=0.001,
        )

        lstm_model, loss_log = train_lstm(
            bb_model,
            lstm_model,
            loss_func,
            device,
            dataloader,
            optimizer,
            models['LSTM']['num_epochs'],
            window_batch,
        )

        loss_log['name'] = 'LSTM with Feature Level Fusion Backbone'

        plot_multiple_losses(
            [loss_log],
            os.path.join(
                RESULTS_PATH,
                f'{USER} {time} LSTM_Losses_Patient_{patient}.png',
            ),
            f'LSTM Classifier Patient Out {patient}',
        )

        torch.save(
            bb_model.state_dict(),
            os.path.join(
                TRAINED_MODELS_PATH,
                'Patient KFold',
                f'{time} Backbone Model Patient Out {patient}.pth',
            ),
        )

        torch.save(
            lstm_model.state_dict(),
            os.path.join(
                TRAINED_MODELS_PATH,
                'Patient KFold',
                f'{time} LSTM Model Patient Out {patient}.pth',
            ),
        )

        data.is_test = True
        dataloader = create_dataloader(data, 1)

        preds, target_labels = test_patient_kfold(
            data, dataloader, bb_model, lstm_model, device)

        best_thr, best_fpr, best_tpr, thr, fpr, tpr = compute_train_roc(
            preds,
            target_labels,
            f'Patient Out {patient}',
            show=True,
        )

        acc = calculate_accuracy(preds, target_labels, best_thr)
        metrics.append((best_thr, best_fpr, best_tpr, acc))
        echo(metrics)
        roc_auc = auc(fpr, tpr)
        roc_curves.append((fpr, tpr, roc_auc))

        del bb_model, lstm_model
        gc.collect()
        torch.cuda.empty_cache()

    plot_roc_curves(roc_curves)
    kfold_boxplot(metrics, 'Patient KFold Boxplots', 'Patient KFold Boxplots')


def test_patient_kfold(data, dataloader, bb_model, lstm_model, device):
    bb_model.eval()
    lstm_model.eval()
    preds = []
    target_labels = []
    for idx, (windows, targets) in enumerate(dataloader):
        windows = windows.squeeze(0)

        windows = windows.to(device)
        windows = bb_model.get_embeddings(windows)
        hn = None
        cn = None
        targets = targets.squeeze(0)
        output, _, _ = lstm_model(windows, hn, cn)
        prob = F.softmax(output, dim=1)

        prob = prob[:, 1]

        prob = prob.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        preds += list(prob)
        target_labels += list(targets)

    return preds, target_labels


def compute_train_roc(
        probs: list[float],
        target_labels: list[int],
        name,
        show=False,
):
    fpr, tpr, thr = roc_curve(target_labels, probs)
    if show:
        plot_roc(fpr, tpr, name)
    best_threshold, best_fpr, best_tpr = get_best_thr(fpr, tpr, thr)

    return best_threshold, best_fpr, best_tpr, thr, fpr, tpr


def get_best_thr(
        false_positive_rates: np.ndarray,
        true_positive_rates: np.ndarray,
        thresholds: np.ndarray,
) -> tuple[float, float, float]:
    best_thr = None
    min_distance = sys.maxsize
    for fpr, tpr, thr in zip(false_positive_rates, true_positive_rates, thresholds):
        dist = dist_thr(fpr, tpr)
        if dist < min_distance:
            min_distance = dist
            best_thr = thr
            best_fpr = fpr
            best_tpr = tpr
    return best_thr, best_fpr, best_tpr


def dist_thr(false_positive_rate: float, true_positive_rate: float) -> float:
    dist = pow(pow(false_positive_rate, 2) + pow(1-true_positive_rate, 2), 0.5)
    return dist


def plot_roc(
        false_positive_rates: np.ndarray,
        true_positive_rates: np.ndarray,
        name,
):
    roc_auc = auc(false_positive_rates, true_positive_rates)
    # Plot the ROC curve
    plt.figure()
    plt.plot(false_positive_rates, true_positive_rates,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Epilepsy LSTM with Feature Level Fusion Backbone')
    plt.legend()
    plt.savefig(os.path.join(
        RESULTS_PATH,
        'Patient KFold',
        f'ROC Curve Fold {name}.png',
    ))


def plot_roc_curves(roc_curves):
    plt.figure(figsize=(10, 8))
    for i, (fpr, tpr, roc_auc) in enumerate(roc_curves):
        plt.plot(fpr, tpr, label=f'Fold {i+1} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--',
             color='gray', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Across K-Folds')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.6, linestyle='--')
    plt.savefig(os.path.join(
        RESULTS_PATH,
        'Patient KFold',
        'KFold ROC Curves.png',
    ))
    plt.show()


def kfold_boxplot(metrics: list[tuple[float]], title_1: str, file_name: str):
    metric_1, metric_2, metric_3, metric_4 = zip(*metrics)
    fig, axes = plt.subplots(2, 2, figsize=(15, 5))  # 1 fila, 3 columnas

    # Crear boxplots individuales
    axes[0, 0].boxplot(metric_1, patch_artist=True)
    axes[0, 0].set_title(title_1)
    axes[0, 0].set_ylabel('Threshold')

    axes[0, 1].boxplot(metric_2, patch_artist=True)
    axes[0, 1].set_title('False Positive Rate')

    axes[1, 0].boxplot(metric_3, patch_artist=True)
    axes[1, 0].set_title('True Positive Rate')

    axes[1, 1].boxplot(metric_4, patch_artist=True)
    axes[1, 1].set_title('Accuracy')

    # Ajustar el espacio entre subplots
    plt.tight_layout()

    plt.savefig(os.path.join(
        RESULTS_PATH,
        'Patient KFold',
        f'{file_name}.png',
    ))


def calculate_accuracy(probabilities, target_labels, threshold=0.5):
    """
    Calculate accuracy from probabilities and target labels.

    Parameters:
    -----------
    probabilities : array-like
        Predicted probabilities of belonging to the positive class.
    target_labels : array-like
        Ground truth binary labels (0 or 1).
    threshold : float, optional
        Threshold to convert probabilities to binary predictions. Default is 0.5.

    Returns:
    --------
    accuracy : float
        Accuracy of the predictions.
    """
    # Convert probabilities to binary predictions
    predictions = (np.array(probabilities) >= threshold).astype(int)

    # Compare predictions with target labels
    correct_predictions = (predictions == np.array(target_labels)).sum()

    # Calculate accuracy
    accuracy = correct_predictions / len(target_labels)

    return accuracy
