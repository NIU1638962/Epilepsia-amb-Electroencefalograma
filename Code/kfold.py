# -*- coding: utf-8 -*- noqa
"""
Created on Thu Jan 9 18:22:30 2024

@author: Joan
"""
import os
import gc

from datetime import datetime, timezone
from typing import List, Tuple

import torch

from datasets import SimpleSeizuresDataset
from dataloaders import create_dataloader
from environ import RESULTS_PATH, TRAINED_MODELS_PATH, USER, PICKLE_PATH
from train import train_classifier, train_lstm
from utils import echo, plot_multiple_losses
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F
import pickle

time = datetime.now(timezone.utc).strftime('%Y-%m-%d--%H-%M--%Z')


def backbones_model_kfold(
        data,
        model,
        loss_func,
        batch_size,
        device,
        num_splits,
        saved_models=False,
):
    SUB_FOLDER = 'Backbones (Windows KFold)'
    metrics = []
    roc_curves = []

    stratified_kfold = StratifiedKFold(
        n_splits=num_splits,
        shuffle=True,
        random_state=10,
    )

    for i, (train_index, test_index) in enumerate(stratified_kfold.split(data['windows'], data['classes'])):
        echo('')
        echo(f'Fold: {i + 1}')

        dataloader_training = create_dataloader(
            SimpleSeizuresDataset(
                data['windows'][train_index],
                data['classes'][train_index],
            ),
            batch_size,
        )

        bb_model = model['model']()

        bb_model.to(device)

        if saved_models:
            bb_model.load_state_dict(torch.load(
                os.path.join(
                    TRAINED_MODELS_PATH,
                    SUB_FOLDER,
                    f'Model {model["model_type"]} Backbone'
                    + f' KFold {i + 1:02d}.pth',
                ),
            ))

        else:
            echo('TRAINING BACKBONE')

            optimizer_backbone = model['optimizer'](
                bb_model.parameters(),
                lr=0.001,
            )

            bb_model, loss_log = train_classifier(
                bb_model,
                loss_func,
                device,
                dataloader_training,
                optimizer_backbone,
                model['num_epochs'],
            )

            loss_log['name'] = model['model_type']

            with open(
                os.path.join(
                    PICKLE_PATH,
                    SUB_FOLDER,
                    f'{USER} {time}'
                    + f' Loss {model["model_type"]} Backbone'
                    + f' KFold {i + 1:02d}.pickle',
                ),
                'wb',
            ) as file:
                pickle.dump(loss_log, file)

            plot_multiple_losses(
                [loss_log],
                os.path.join(
                    RESULTS_PATH,
                    SUB_FOLDER,
                    f'{USER} {time}'
                    + f' Loss {model["model_type"]} Backbone'
                    + f' KFold {i + 1:02d}.png',
                ),
                f'Backbone Classifier ({batch_size}) Fold: {i + 1}',
            )

            torch.save(
                bb_model.state_dict(),
                os.path.join(
                    TRAINED_MODELS_PATH,
                    SUB_FOLDER,
                    f'{USER} {time}'
                    + f' Model {model["model_type"]} Backbone'
                    + f' KFold {i + 1:02d}.pth',
                ),
            )

            del dataloader_training, optimizer_backbone
            gc.collect()
            torch.cuda.empty_cache()

        echo('TESTING BACKBONE')

        dataloader_testing = create_dataloader(
            SimpleSeizuresDataset(
                data['windows'][test_index],
                data['classes'][test_index],
            ),
            batch_size,
        )

        predictions, target_labels = test_model_backbone(
            dataloader_testing,
            bb_model,
            device,
        )

        best_thr, best_fpr, best_tpr, thr, fpr, tpr = compute_train_roc(
            predictions,
            target_labels,
            f'ROC Curve of {model["model_type"]} Backbone'
            + f'\nFold: {i + 1}',
            os.path.join(
                RESULTS_PATH,
                SUB_FOLDER,
                f'{USER} {time}'
                + f' ROC Curve of {model["model_type"]} Backbone'
                + f' Fold: {i + 1:02d}.png',
            ),
            show=True,
        )

        accuracy = calculate_accuracy(predictions, target_labels, best_thr)

        metrics.append((best_thr, best_fpr, best_tpr, accuracy))

        echo(
            f'Best Threshold: {metrics[-1][0]:.10f}'
            + f', False Positive Rate: {metrics[-1][1]:.10f}'
            + f', True Positive Rate: {metrics[-1][2]:.10f}'
            + f', Accuracy: {metrics[-1][3]:.10f}'
        )

        roc_auc = auc(fpr, tpr)

        roc_curves.append((fpr, tpr, roc_auc))

        del bb_model, dataloader_testing
        gc.collect()
        torch.cuda.empty_cache()

    metrics_stats = mean_kfold(metrics)
    echo(
        f'Best Threshold: {metrics_stats[0][0]:.10f}'
        + f' ±{metrics_stats[0][1]:.10f}'
        + f', False Positive Rate: {metrics_stats[1][0]:.10f}'
        + f' ±{metrics_stats[1][1]:.10f}'
        + f', True Positive Rate: {metrics_stats[2][0]:.10f}'
        + f' ±{metrics_stats[2][1]:.10f}'
        + f', Accuracy: {metrics_stats[3][0]:.10f}'
        + f' ±{metrics_stats[3][1]:.10f}'
    )
    plot_roc_curves(
        roc_curves,
        'Fold',
        f'ROC Curves Across K-Folds of {model["model_type"]} Backbone',
        os.path.join(
            RESULTS_PATH,
            SUB_FOLDER,
            f'{USER} {time} '
            + f'ROC Curves Across K-Folds of {model["model_type"]} Backbone.png',
        ),
    )

    with open(
        os.path.join(
            PICKLE_PATH,
            SUB_FOLDER,
            f'{USER} {time}'
            + f' Metrics of {model["model_type"]} Backbone.pickle',
        ),
        'wb',
    ) as file:
        pickle.dump(metrics, file)

    with open(
        os.path.join(
            PICKLE_PATH,
            SUB_FOLDER,
            f'{USER} {time}'
            + f' Statistical Metrics of {model["model_type"]} Backbone.pickle',
        ),
        'wb',
    ) as file:
        pickle.dump(metrics_stats, file)


def generalized_model_patient_kfold(
        data,
        models,
        loss_func,
        batch_size,
        window_batch,
        device,
        model_params,
        saved_models=False,
):
    SUB_FOLDER = 'Generalized Model (Patient KFold)'
    num_patients = data.num_patients

    patients = np.array([i for i in range(num_patients)])

    roc_curves = []
    metrics = []

    data.is_personalized = False
    data.test_recording = None

    for patient in patients:
        echo('')
        echo(f'Patient Out: {patient + 1}')

        data.is_test = False
        data.is_lstm = False
        data.patient = patient

        bb_model = models['BB']['model']()

        bb_model.to(device)

        if saved_models:
            bb_model.load_state_dict(torch.load(
                os.path.join(
                    TRAINED_MODELS_PATH,
                    SUB_FOLDER,
                    'Model Feature Level Fusion Backbone'
                    + f' Patient Out {patient + 1:02d}.pth',
                ),
            ))

        else:
            echo('TRAINING FEATURE LEVEL FUSION BACKBONE')

            dataloader_training_backbone = create_dataloader(data, batch_size)

            optimizer_backbone = models['BB']['optimizer'](
                bb_model.parameters(),
                lr=0.001,
            )

            bb_model, loss_log = train_classifier(
                bb_model,
                loss_func,
                device,
                dataloader_training_backbone,
                optimizer_backbone,
                models['BB']['num_epochs'],
            )

            loss_log['name'] = 'Feature Level Fusion Backbone'

            with open(
                os.path.join(
                    PICKLE_PATH,
                    SUB_FOLDER,
                    f'{USER} {time}'
                    + ' Loss Feature Level Fusion Backbone'
                    + f' Patient Out {patient + 1:02d}.pickle',
                ),
                'wb',
            ) as file:
                pickle.dump(loss_log, file)

            plot_multiple_losses(
                [loss_log],
                os.path.join(
                    RESULTS_PATH,
                    SUB_FOLDER,
                    f'{USER} {time}'
                    + ' Loss Feature Level Fusion Backbone'
                    + f' Patient Out {patient + 1:02d}.png',
                ),
                f'Backbone Classifier Patient Out: {patient + 1}',
            )

            torch.save(
                bb_model.state_dict(),
                os.path.join(
                    TRAINED_MODELS_PATH,
                    SUB_FOLDER,
                    f'{USER} {time}'
                    + ' Model Feature Level Fusion Backbone'
                    + f' Patient Out {patient + 1:02d}.pth',
                ),
            )

            del dataloader_training_backbone, optimizer_backbone
            gc.collect()
            torch.cuda.empty_cache()

        bb_model.eval()

        data.is_lstm = True

        lstm_model = models['LSTM']['model'](*model_params)
        lstm_model.to(device)

        if saved_models:
            lstm_model.load_state_dict(torch.load(
                os.path.join(
                    TRAINED_MODELS_PATH,
                    SUB_FOLDER,
                    'Model LSTM with Feature Level Fusion Backbone'
                    + f' Patient Out {patient + 1:02d}.pth',
                ),
            ))

        else:
            echo('TRAINING LSTM WITH FEATURE LEVEL FUSION BACKBONE')

            dataloader_training_lstm = create_dataloader(data, 1)

            optimizer_lstm = models['LSTM']['optimizer'](
                lstm_model.parameters(),
                lr=0.001,
            )

            lstm_model, loss_log = train_lstm(
                bb_model,
                lstm_model,
                loss_func,
                device,
                dataloader_training_lstm,
                optimizer_lstm,
                models['LSTM']['num_epochs'],
                window_batch,
            )

            loss_log['name'] = 'LSTM with Feature Level Fusion Backbone'

            with open(
                os.path.join(
                    PICKLE_PATH,
                    SUB_FOLDER,
                    f'{USER} {time}'
                    + ' Loss LSTM with Feature Level Fusion Backbone'
                    + f' Patient Out {patient + 1:02d}.pickle',
                ),
                'wb',
            ) as file:
                pickle.dump(loss_log, file)

            plot_multiple_losses(
                [loss_log],
                os.path.join(
                    RESULTS_PATH,
                    SUB_FOLDER,
                    f'{USER} {time}'
                    + ' Loss LSTM with Feature Level Fusion Backbone'
                    + f' Patient Out {patient + 1:02d}.png',
                ),
                f'LSTM Classifier Patient Out {patient + 1}',
            )

            torch.save(
                lstm_model.state_dict(),
                os.path.join(
                    TRAINED_MODELS_PATH,
                    SUB_FOLDER,
                    f'{USER} {time}'
                    + ' Model LSTM with Feature Level Fusion Backbone'
                    + f' Patient Out {patient + 1:02d}.pth',
                ),
            )

            del dataloader_training_lstm, optimizer_lstm
            gc.collect()
            torch.cuda.empty_cache()

        echo('TESTING LSTM WITH FEATURE LEVEL FUSION BACKBONE')

        data.is_test = True
        dataloader_testing = create_dataloader(data, 1)

        predictions, target_labels = test_model_kfold(
            data,
            dataloader_testing,
            bb_model,
            lstm_model,
            device,
        )

        best_thr, best_fpr, best_tpr, thr, fpr, tpr = compute_train_roc(
            predictions,
            target_labels,
            'ROC Curve of Generalized LSTM with Feature Level Fusion Backbone'
            + f'\nFold with Patient Out: {patient + 1}',
            os.path.join(
                RESULTS_PATH,
                'Generalized Model (Patient KFold)',
                f'{USER} {time}'
                + ' ROC Curve LSTM with Feature Level Fusion Backbone'
                + f' Fold with Patient Out {patient + 1:02d}.png',
            ),
            show=True,
        )

        accuracy = calculate_accuracy(predictions, target_labels, best_thr)

        metrics.append((best_thr, best_fpr, best_tpr, accuracy))

        echo(
            f'Best Threshold: {metrics[-1][0]:.10f}'
            + f', False Positive Rate: {metrics[-1][1]:.10f}'
            + f', True Positive Rate: {metrics[-1][2]:.10f}'
            + f', Accuracy: {metrics[-1][3]:.10f}'
        )

        roc_auc = auc(fpr, tpr)

        roc_curves.append((fpr, tpr, roc_auc))

        del bb_model, dataloader_testing, lstm_model
        gc.collect()
        torch.cuda.empty_cache()

    metrics_stats = mean_kfold(metrics)

    echo(
        f'Best Threshold: {metrics_stats[0][0]:.10f}'
        + f' ±{metrics_stats[0][1]:.10f}'
        + f', False Positive Rate: {metrics_stats[1][0]:.10f}'
        + f' ±{metrics_stats[1][1]:.10f}'
        + f', True Positive Rate: {metrics_stats[2][0]:.10f}'
        + f' ±{metrics_stats[2][1]:.10f}'
        + f', Accuracy: {metrics_stats[3][0]:.10f}'
        + f' ±{metrics_stats[3][1]:.10f}'
    )

    plot_roc_curves(
        roc_curves,
        'Fold with Patient Out',
        'ROC Curves Across K-Folds for Generalized Model',
        os.path.join(
            RESULTS_PATH,
            SUB_FOLDER,
            f'{USER} {time} ROC Curves Across K-Folds.png',
        ),
    )

    with open(
        os.path.join(
            PICKLE_PATH,
            SUB_FOLDER,
            f'{USER} {time} Metrics.pickle'
        ),
        'wb',
    ) as file:
        pickle.dump(metrics, file)

    with open(
        os.path.join(
            PICKLE_PATH,
            SUB_FOLDER,
            f'{USER} {time} Statistical Metrics.pickle'
        ),
        'wb',
    ) as file:
        pickle.dump(metrics_stats, file)


def personalized_model_record_kfold(
        data,
        models,
        loss_func,
        batch_size,
        window_batch,
        device,
        model_params,
        saved_models=False,
        plot_generalized = False
):
    SUB_FOLDER = 'Personalized Model (Recording KFold)'
    num_patients = data.num_patients
    patients = np.array([i for i in range(num_patients)])
    data.is_personalized = True
    data.is_test = False
    data.test_recording = None

    for patient in patients:
        metrics = []
        roc_curves = []
        data.test_recording = None
        echo('')
        echo(f'Model for Patient: {patient + 1}')
        data.patient = patient
        num_recordings = data.len_patient_recordings
        recordings = np.array([i for i in range(num_recordings)])

        for recording in recordings:
            echo('')
            echo(f'Recording Out: {recording + 1} / {num_recordings}')
            data.is_test = False
            data.is_lstm = False
            data.test_recording = recording
            bb_model = models['BB']['model']()

            bb_model.to(device)

            if saved_models:
                bb_model.load_state_dict(torch.load(
                    os.path.join(
                        TRAINED_MODELS_PATH,
                        SUB_FOLDER,
                        'Model Feature Level Fusion Backbone'
                        + f' Patient {patient + 1:02d}'
                        + f' Recording {recording + 1:02d}.pth',
                    ),
                ))
            else:
                echo('TRAINING FEATURE LEVEL FUSION BACKBONE')

                dataloader_training_backbone = create_dataloader(
                    data,
                    batch_size,
                )

                optimizer_backbone = models['BB']['optimizer'](
                    bb_model.parameters(), lr=0.001)

                bb_model, loss_log = train_classifier(
                    bb_model,
                    loss_func,
                    device,
                    dataloader_training_backbone,
                    optimizer_backbone,
                    models['BB']['num_epochs'],
                )

                loss_log['name'] = 'Feature Level Fusion Backbone'

                with open(
                    os.path.join(
                        PICKLE_PATH,
                        SUB_FOLDER,
                        'Loss Feature Level Fusion Backbone'
                        + f' Patient {patient + 1:02d}'
                        + f' Recording {recording + 1:02d}.pickle'
                    ),
                    'wb',
                ) as file:
                    pickle.dump(loss_log, file)

                plot_multiple_losses(
                    [loss_log],
                    os.path.join(
                        RESULTS_PATH,
                        SUB_FOLDER,
                        f'{USER} {time}'
                        + ' Loss Feature Level Fusion Backbone'
                        + f' Patient {patient + 1:02d}'
                        + f' Recording {recording + 1:02d}.png',
                    ),
                    'Backbone for'
                    + f'Patient {patient + 1} for Recording {recording + 1}',
                )

                torch.save(
                    bb_model.state_dict(),
                    os.path.join(
                        TRAINED_MODELS_PATH,
                        SUB_FOLDER,
                        f'{USER} {time}'
                        + ' Model Feature Level Fusion Backbone'
                        + f' Patient {patient + 1:02d}'
                        + f' Recording {recording + 1:02d}.pth',
                    ),
                )

                del dataloader_training_backbone, optimizer_backbone
                gc.collect()
                torch.cuda.empty_cache()

            bb_model.eval()

            data.is_lstm = True

            lstm_model = models['LSTM']['model'](*model_params)
            lstm_model.to(device)

            if saved_models:
                lstm_model.load_state_dict(torch.load(
                    os.path.join(
                        TRAINED_MODELS_PATH,
                        SUB_FOLDER,
                        'Model LSTM with Feature Level Fusion Backbone for'
                        + f' Patient {patient + 1:02d}'
                        + f' Recording {recording + 1:02d}.pth',
                    ),
                ))

            else:
                echo('TRAINING LSTM WITH FEATURE LEVEL FUSION BACKBONE')

                dataloader_training_lstm = create_dataloader(data, 1)

                optimizer_lstm = models['LSTM']['optimizer'](
                    lstm_model.parameters(),
                    lr=0.001,
                )

                lstm_model, loss_log = train_lstm(
                    bb_model,
                    lstm_model,
                    loss_func,
                    device,
                    dataloader_training_lstm,
                    optimizer_lstm,
                    models['LSTM']['num_epochs'],
                    window_batch,
                )

                loss_log['name'] = 'LSTM with Feature Level Fusion Backbone'

                with open(
                    os.path.join(
                        PICKLE_PATH,
                        SUB_FOLDER,
                        f'{USER} {time}'
                        + ' Loss LSTM with Feature Level Fusion Backbone'
                        + f' Patient {patient + 1:02d}'
                        + f' Recording {recording + 1:02d}.pickle'
                    ),
                    'wb',
                ) as file:
                    pickle.dump(loss_log, file)

                plot_multiple_losses(
                    [loss_log],
                    os.path.join(
                        RESULTS_PATH,
                        SUB_FOLDER,
                        f'{USER} {time}'
                        + ' Loss LSTM with Feature Level Fusion Backbone'
                        + f' Patient {patient + 1:02d}'
                        + f' Recording {recording + 1:02d}.png',
                    ),
                    'LSTM with Feature Level Fusion Backbone for'
                    + f' Patient {patient + 1} for'
                    + f' Recording {recording + 1}',
                )

                torch.save(
                    lstm_model.state_dict(),
                    os.path.join(
                        TRAINED_MODELS_PATH,
                        SUB_FOLDER,
                        f'{USER} {time}'
                        + ' Model LSTM with Feature Level Fusion Backbone for'
                        + f' Patient {patient + 1:02d}'
                        + f' Recording {recording + 1:02d}.pth',
                    ),
                )

                del dataloader_training_lstm, optimizer_lstm
                gc.collect()
                torch.cuda.empty_cache()

            echo('TESTING LSTM WITH FEATURE LEVEL FUSION BACKBONE')

            data.is_test = True
            dataloader_testing = create_dataloader(data, 1)

            predictions, target_labels = test_model_kfold(
                data,
                dataloader_testing,
                bb_model,
                lstm_model,
                device,
            )

            best_thr, best_fpr, best_tpr, thr, fpr, tpr = compute_train_roc(
                predictions,
                target_labels,
                'ROC Curve of Personalized LSTM with Feature Level Fusion'
                + f' Backbone\nfor Patient {patient + 1}'
                + f' Fold with Record Out: {recording + 1}',
                os.path.join(
                    RESULTS_PATH,
                    SUB_FOLDER,
                    f'{USER} {time}'
                    + ' ROC Curve LSTM with Feature Level Fusion Backbone'
                    + f' for Patient {patient + 1:02d}'
                    + f' Fold with Record Out {recording + 1:02d}.png',
                ),
                show=True,
            )

            accuracy = calculate_accuracy(predictions, target_labels, best_thr)

            metrics.append((best_thr, best_fpr, best_tpr, accuracy))

            echo(
                f'Best Threshold: {metrics[-1][0]:.10f}'
                + f', False Positive Rate: {metrics[-1][1]:.10f}'
                + f', True Positive Rate: {metrics[-1][2]:.10f}'
                + f', Accuracy: {metrics[-1][3]:.10f}'
            )
            roc_auc = auc(fpr, tpr)
            roc_curves.append((fpr, tpr, roc_auc))

            del bb_model, dataloader_testing, lstm_model
            gc.collect()
            torch.cuda.empty_cache()

        metrics_stats = mean_kfold(metrics)
        
        if(plot_generalized):
            bb_model_generalized = models['BB']['model']()
            bb_model_generalized.to(device)
            lstm_model_generalized = models['LSTM']['model'](*model_params)
            lstm_model_generalized.to(device)

            bb_model_generalized.load_state_dict(torch.load(
                os.path.join(
                    TRAINED_MODELS_PATH,
                    SUB_FOLDER,
                    'Model Feature Level Fusion Backbone'
                    + f' Patient Out {patient + 1:02d}.pth',
                ),
            ))

            lstm_model_generalized.load_state_dict(torch.load(
                os.path.join(
                    TRAINED_MODELS_PATH,
                    SUB_FOLDER,
                    'Model LSTM with Feature Level Fusion Backbone for'
                    + f' Patient {patient + 1:02d}'
                    + f' Recording {recording + 1:02d}.pth',
                ),
            ))

            predictions_generalized, target_labels_generalized = test_model_kfold(
                data,
                dataloader_testing,
                bb_model_generalized,
                lstm_model_generalized,
                device,
            )

            best_thr_gen, best_fpr_gen, best_tpr_gen, thr_gen, fpr_gen, tpr_gen = compute_train_roc(
                predictions_generalized,
                target_labels_generalized,
                'ROC Curve of Generalized LSTM with Feature Level Fusion Backbone'
                + f'\nFold with Patient Out: {patient + 1}',
                "Resultado",
                show=True,
            )

            roc_auc_gen = auc(fpr_gen, tpr_gen)

        echo(
            f'Best Threshold: {metrics_stats[0][0]:.10f}'
            + f' ±{metrics_stats[0][1]:.10f}'
            + f', False Positive Rate: {metrics_stats[1][0]:.10f}'
            + f' ±{metrics_stats[1][1]:.10f}'
            + f', True Positive Rate: {metrics_stats[2][0]:.10f}'
            + f' ±{metrics_stats[2][1]:.10f}'
            + f', Accuracy: {metrics_stats[3][0]:.10f}'
            + f' ±{metrics_stats[3][1]:.10f}'
        )

        plot_roc_curves(
            roc_curves,
            'Fold with Recording Out',
            'ROC Curves Across K-Folds for Personalized Model'
            + f' for Patient {patient + 1}',
            os.path.join(
                RESULTS_PATH,
                SUB_FOLDER,
                f'{USER} {time} ROC Curves Across K-Folds'
                + f' for Patient {patient + 1:02d}.png',
            ),
            roc_auc_gen,
            
        )

        with open(
            os.path.join(
                PICKLE_PATH,
                SUB_FOLDER,
                f'{USER} {time} Metrics' +
                f' for Patient {patient + 1:02d}.pickle',
            ),
            'wb',
        ) as file:
            pickle.dump(metrics, file)

        with open(
            os.path.join(
                PICKLE_PATH,
                SUB_FOLDER,
                f'{USER} {time} Statistical Metrics'
                + f'for Patient {patient + 1:02d}.pickle',
            ),
            'wb',
        ) as file:
            pickle.dump(metrics_stats, file)


def test_backbone(
        data,
        models,
        loss_func,
        batch_size,
        window_batch,
        device,
        model_params,
):
    SUB_FOLDER = 'Generalized Model (Patient KFold, Only Backbone)'
    num_patients = data.num_patients
    patients = np.array([i for i in range(num_patients)])
    roc_curves = []
    metrics = []
    data.is_personalized = False
    data.test_recording = None
    for patient in patients:
        echo('')
        echo(f'Patient Out: {patient + 1}')
        echo('TESTING FEATURE LEVEL FUSION BACKBONE')
        data.is_test = True
        data.is_lstm = False
        data.patient = patient

        dataloader_testing = create_dataloader(data, batch_size)
        bb_model = models['BB']['model']()
        bb_model.load_state_dict(torch.load(
            os.path.join(
                TRAINED_MODELS_PATH,
                SUB_FOLDER,
                'Model Feature Level Fusion Backbone'
                + f' Patient Out {patient + 1:02d}.pth',
            ),
        ))

        bb_model.to(device)

        predictions, target_labels = test_model_backbone(
            dataloader_testing,
            bb_model,
            device,
        )

        best_thr, best_fpr, best_tpr, thr, fpr, tpr = compute_train_roc(
            predictions,
            target_labels,
            'ROC Curve of Generalized Feature Level Fusion Backbone'
            + f'\nFold with Patient Out: {patient + 1}',
            os.path.join(
                RESULTS_PATH,
                SUB_FOLDER,
                f'{USER} {time}'
                + ' ROC Curve with Feature Level Fusion Backbone'
                + f' Fold with Patient Out {patient + 1:02d}.png',
            ),
            show=True,
        )

        accuracy = calculate_accuracy(predictions, target_labels, best_thr)

        metrics.append((best_thr, best_fpr, best_tpr, accuracy))

        echo(
            f'Best Threshold: {metrics[-1][0]:.10f}'
            + f', False Positive Rate: {metrics[-1][1]:.10f}'
            + f', True Positive Rate: {metrics[-1][2]:.10f}'
            + f', Accuracy: {metrics[-1][3]:.10f}'
        )

        roc_auc = auc(fpr, tpr)

        roc_curves.append((fpr, tpr, roc_auc))

        del bb_model, dataloader_testing
        gc.collect()
        torch.cuda.empty_cache()

    metrics_stats = mean_kfold(metrics)
    echo(
        f'Best Threshold: {metrics_stats[0][0]:.10f}'
        + f' ±{metrics_stats[0][1]:.10f}'
        + f', False Positive Rate: {metrics_stats[1][0]:.10f}'
        + f' ±{metrics_stats[1][1]:.10f}'
        + f', True Positive Rate: {metrics_stats[2][0]:.10f}'
        + f' ±{metrics_stats[2][1]:.10f}'
        + f', Accuracy: {metrics_stats[3][0]:.10f}'
        + f' ±{metrics_stats[3][1]:.10f}'
    )
    plot_roc_curves(
        roc_curves,
        'Fold with Patient Out',
        'ROC Curves Across K-Folds for Generalized Model (Only Backbone)',
        os.path.join(
            RESULTS_PATH,
            SUB_FOLDER,
            f'{USER} {time} ROC Curves Across K-Folds.png',
        ),
    )


def test_model_backbone(
        dataloader,
        bb_model,
        device,
):

    bb_model.eval()
    preds = []
    target_labels = []
    for idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)

        outputs = bb_model(inputs)

        prob = F.softmax(outputs, dim=1)

        prob = prob[:, 1]

        preds += list(prob.cpu().detach().numpy())
        target_labels += list(targets.cpu().detach().numpy())

        del inputs, targets, prob, outputs
        gc.collect()
        torch.cuda.empty_cache()
    return preds, target_labels


def test_model_kfold(
        data,
        dataloader,
        bb_model,
        lstm_model,
        device,
):
    bb_model.eval()
    lstm_model.eval()
    preds = []
    target_labels = []
    for idx, (windows, targets) in enumerate(dataloader):
        windows = windows.squeeze(0)

        echo(windows.shape)

        windows = windows.to(device)

        with torch.no_grad():
            windows = bb_model.get_embeddings(windows)
            hn = None
            cn = None
            targets = targets.squeeze(0)
            output, _, _ = lstm_model(windows, hn, cn)
            prob = F.softmax(output, dim=1)

            prob = prob[:, 1]

            preds += list(prob.cpu().detach().numpy())
            target_labels += list(targets.cpu().detach().numpy())

        del windows, targets, prob, output, _
        gc.collect()
        torch.cuda.empty_cache()

    free_memory, total_memory = torch.cuda.mem_get_info(
        torch.cuda.current_device())

    used_memory = (total_memory - free_memory)

    free_memory_mb = free_memory / 1024 ** 2
    total_memory_mb = total_memory / 1024 ** 2
    used_memory_mb = used_memory / 1024 ** 2

    echo(
        f'Total Memory: {total_memory_mb}MB'
        + f', Used Memory: {used_memory_mb}:MB'
        + f', Free Memory: {free_memory_mb}MB'
    )

    return preds, target_labels


def compute_train_roc(
        probs: List[float],
        target_labels: List[int],
        title: str,
        file_name: str,
        show=False,
):
    fpr, tpr, thr = roc_curve(target_labels, probs)
    if show:
        plot_roc(fpr, tpr, title, file_name)
    best_threshold, best_fpr, best_tpr = get_best_thr(fpr, tpr, thr)

    return best_threshold, best_fpr, best_tpr, thr, fpr, tpr


def get_best_thr(
        false_positive_rates: np.ndarray,
        true_positive_rates: np.ndarray,
        thresholds: np.ndarray,
) -> Tuple[float, float, float]:
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
        title: str,
        file_name: str,
):
    roc_auc = auc(false_positive_rates, true_positive_rates)
    # Plot the ROC curve
    plt.figure()
    plt.plot(false_positive_rates, true_positive_rates,
             label=f'ROC curve (area = {roc_auc:.5f})')
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.savefig(file_name)
    plt.close()


def plot_roc_curves(roc_curves, label_title, title, file_name, generalized_roc_curve = None):
    plt.figure(figsize=(10, 8))
    for i, (fpr, tpr, roc_auc) in enumerate(roc_curves):
        plt.plot(
            fpr,
            tpr,
            label=f'{label_title}: {i+1} (AUC = {roc_auc:.5f})',
        )

    if(generalized_roc_curve != None):
        plt.plot(
            generalized_roc_curve[0],
            generalized_roc_curve[1],
            label=f'{label_title}: {i+1} (AUC = {generalized_roc_curve[2]:.5f})',
            linewidth=3
        )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle='--',
        color='gray',
        label='Random Guess',
    )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.6, linestyle='--')
    plt.savefig(file_name)
    # plt.show()
    plt.close()


def kfold_boxplot(metrics: List[Tuple[float]], title_1: str, file_name: str):
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
        'Generalized Model (Patient KFold)',
        f'{file_name}.png',
    ))
    plt.close()


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


def mean_kfold(metrics: List[Tuple[float]]) -> List[Tuple[float, float]]:
    metric_1, metric_2, metric_3, metric_4 = zip(*metrics)
    metrics = [metric_1, metric_2, metric_3, metric_4]
    metrics_stats = []
    for metric in metrics:
        media = sum(metric) / len(metric)

        # Varianza
        varianza = sum((x - media) ** 2 for x in metric) / len(metric)

        # Desviación estándar poblacional
        desviacion = varianza ** 0.5
        metrics_stats.append((media, desviacion))
    return metrics_stats


def gen_personalized_boxplots():
    pickle_files = os.listdir(os.path.join(
        PICKLE_PATH, 'Generalized Model (Patient KFold)'))
    file = sorted([file for file in pickle_files if file.startswith(
        'Metrics')])

    with open(os.path.join(
            PICKLE_PATH,
            'Generalized Model (Patient KFold)',
            file[0]), 'rb') as metrics:

        gen_metric_values = pickle.load(metrics)

    pickle_files = os.listdir(os.path.join(
        PICKLE_PATH, 'Personalized Model (Recording KFold)'))
    files = sorted([file for file in pickle_files if file.startswith(
        'Metrics')])

    data = []
    for file in files:
        with open(os.path.join(
                PICKLE_PATH,
                'Personalized Model (Recording KFold)',
                file), 'rb') as metrics:

            data.append(pickle.load(metrics))

    num_metrics = len(data[0][0])
    metric_values = {i: [] for i in range(num_metrics)}

    for patient in data:
        for i in range(num_metrics):
            metric_values[i].append([recording[i] for recording in patient])

    names = ['Threshold', 'False Positive Rate',
             'True Positive Rate', 'Accuracy']
    for (metric_idx, patient_values), gen_values, name in zip(metric_values.items(), zip(*gen_metric_values), names):
        # Flaten los valores por paciente para cada métrica
        patient_metrics = [np.array(values).flatten()
                           for values in patient_values]
        # Crear boxplot
        plt.figure(figsize=(8, 6))

        plt.gcf().canvas.setWindowTitle(name + "Boxplots for Perzonalized")
        boxplot = plt.boxplot(patient_metrics, labels=[
            f"{i+1}" for i in range(len(patient_metrics))], patch_artist=True)

        for i, box in enumerate(boxplot['boxes']):
            box.set_facecolor('skyblue')

            plt.text(i+1, -0.09, f'({len(data[i])})',
                     ha='center', va='bottom', fontsize=10, color='black')

        for i, point in enumerate(gen_values):
            plt.scatter(i+1, point, color='red',
                        label='Generalized' if i == 0 else "", zorder=3, marker='D')

        for x in range(1, len(patient_metrics) + 1):
            plt.axvline(x=x, color='gray', linestyle='--',
                        linewidth=0.5, zorder=1)

        plt.ylim((0, 1))
        plt.title(f'{name}', fontweight='bold')
        plt.xlabel('Patients (N recordings)', labelpad=15, fontweight='bold')
        plt.ylabel(name+' Values', fontweight='bold')
        plt.grid(axis='y')
        plt.xticks(fontweight='bold')

        plt.legend(loc='upper right', fontsize=10)

        plt.show()


def gen_gen_v_per_boxplots():
    pickle_files = os.listdir(os.path.join(
        PICKLE_PATH, 'Generalized Model (Patient KFold)'))
    file = sorted([file for file in pickle_files if file.startswith(
        'Metrics')])

    with open(os.path.join(
            PICKLE_PATH,
            'Generalized Model (Patient KFold)',
            file[0]), 'rb') as metrics:

        gen_metric_values = pickle.load(metrics)

    pickle_files = os.listdir(os.path.join(
        PICKLE_PATH, 'Personalized Model (Recording KFold)'))
    files = sorted([file for file in pickle_files if file.startswith(
        'Metrics')])

    data_per = []
    for file in files:
        with open(os.path.join(
                PICKLE_PATH,
                'Personalized Model (Recording KFold)',
                file), 'rb') as metrics:

            data_per.append(pickle.load(metrics))

    per_metric_values = [tup for sublist in data_per for tup in sublist]

    names = ['Threshold', 'False Positive Rate',
             'True Positive Rate', 'Accuracy']
    labels = ['Generalized', 'Personalized']

    for general_metrics, patient_metrics, name in zip(zip(*gen_metric_values), zip(*per_metric_values), names):
        metrics = [general_metrics, patient_metrics]

        if name == 'Threshold':
            # Crear violinplot
            plt.figure(figsize=(8, 6))
            plt.violinplot(metrics)
            plt.xticks([])

            positions = range(1, len(metrics) + 1)
            for pos, label in zip(positions, labels):
                plt.text(pos, -0.09, label, ha='center',
                         va='bottom', fontsize=10, fontweight='bold')

        else:
            # Crear boxplot
            plt.figure(figsize=(8, 6))
            boxplot = plt.boxplot(metrics, labels=labels, patch_artist=True)

            for i, box in enumerate(boxplot['boxes']):
                box.set_facecolor('skyblue')

                # plt.text(i+1, -0.09, f'({len(data[i])})',
                #         ha='center', va='bottom', fontsize=10, color='black')

        plt.ylim((0, 1))
        plt.title(f'{name}', fontweight='bold')
        plt.xlabel('Patients (N recordings)', labelpad=15, fontweight='bold')
        plt.ylabel(name+' Values', fontweight='bold')
        plt.grid(axis='y')
        plt.xticks(fontweight='bold')
        plt.show()
# gen_personalized_boxplots()
# gen_gen_v_per_boxplots()
