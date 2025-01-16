# -*- coding: utf-8 -*- noqa
"""
Created on Thu Jan 16 00:33:50 2025

@author: Joel Tapia Salvador
"""
from datasets import SeizuresDataset
from dataloaders import create_dataloader
from environ import DATA_PATH, PICKLE_PATH

import pickle
import os

if __name__ == "__main__":
    distribution = []
    dataset = SeizuresDataset(DATA_PATH)

    dataset.is_lstm = True
    dataset.is_test = True

    for patient in range(dataset.num_patients):
        dataset.patient = patient
        distribution.append([])

        for recording in range(dataset.len_patient_recordings):
            dataset.test_recording = recording
            distribution[patient].append([])
            dataloader = create_dataloader(dataset, 1)

            for _, targets in dataloader:
                del _
                num_0 = 0
                num_1 = 1
                for target in targets:
                    if target == 0:
                        num_0 += 1
                    elif target == 1:
                        num_1 += 1

                distribution[patient][recording].append(num_0)
                distribution[patient][recording].append(num_1)

    with open(os.path.join(PICKLE_PATH, 'distribution.pickle'), 'wb') as file:
        pickle.dump(file)
