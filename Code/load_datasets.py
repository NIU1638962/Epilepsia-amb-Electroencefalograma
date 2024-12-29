# -*- coding: utf-8 -*- noqa
"""
Created on Fri Dec 27 19:31:14 2024

@author: Sergio
"""
import os
import re
import sys

import numpy as np
import pandas as pd

from typing import Tuple

from environ import DATA_PATH, DEBUG
from utils import echo


def load_seizures(
        path_root_directory: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset with seizures.

    Parameters
    ----------
    path_root_directory : string
        Path to dataset.

    Returns
    -------
    windows : Numpy array
        All windows of all patients concatenated in the same array.
    classes : Numpy array
        Array with the class of the windows. Is a mask 1 on 1 to the "windows"
        array.
    patients_ids : Numpy array
        Array with the patient if of the windows. Is a mask 1 on 1 to the
        "windows" array.
    recordings : Numpy array
        Array with the recording session of a patient of the window. Is a mask
        1 on 1 to the "windows" array.

    """
    files = os.listdir(path_root_directory)
    files = sorted([file for file in files if file.endswith(
        ('.npz', '.parquet'))])
    if DEBUG:
        echo(files)

    separadores = r'[._]'
    recordings = []
    for i in range(0, len(files), 2):
        if DEBUG:
            echo(files[i], files[i+1])

        patient_df = pd.read_parquet(os.path.join(path_root_directory,files[i+1]))
        patient_data_np = np.load(
            os.path.join(path_root_directory,files[i]), allow_pickle=True)

        patient_id = files[i].split('_')[0][-2:]
        filenames_array = patient_df['filename'].to_numpy()

        patient_windows = patient_data_np['EEG_win'].astype(np.float32)

        patient_classes = patient_df['class'].to_numpy(dtype=np.int64)
        patient_id_array = np.full(
            (patient_windows.shape[0],), int(patient_id), dtype=np.int8)
        if 'windows' not in locals():     # Primera iteración
            windows = patient_windows
            classes = patient_classes
            patients_ids = patient_id_array

        else:
            windows = np.vstack((windows, patient_windows))
            classes = np.hstack((classes, patient_classes))
            patients_ids = np.hstack((patients_ids, patient_id_array))

        for i in range(filenames_array.shape[0]):
            record = re.split(separadores, filenames_array[i])[1]
            recordings.append(record)

        if DEBUG:
            echo(windows.shape)
            echo(classes.shape)
            echo(patients_ids.shape)
            echo(len(recordings))

        assert (
            (
                windows.shape[0] == classes.shape[0]
            ) and (
                windows.shape[0] == patients_ids.shape[0]
            ) and (
                windows.shape[0] == len(recordings)
            )
        ), 'Loaded data arrays have not the same 0th dimension shape.'

    recordings = np.array(recordings)  # Esto es un array de strings

    assert (
        (
            windows.shape[0] == classes.shape[0]
        ) and (
            windows.shape[0] == patients_ids.shape[0]
        ) and (
            windows.shape[0] == recordings.shape[0]
        )
    ), 'Loaded data arrays have not the same 0th dimension shape.'

    return windows, classes, patients_ids, recordings


if __name__ == '__main__':
    windows, classes, patients_ids, recordings = load_seizures(DATA_PATH)
    if DEBUG:
        echo('Loaded:')
        echo(windows.shape, str(sys.getsizeof(windows)))
        echo(classes.shape, str(sys.getsizeof(classes)))
        echo(patients_ids.shape, str(sys.getsizeof(patients_ids)))
        echo(recordings.shape, str(sys.getsizeof(recordings)))
