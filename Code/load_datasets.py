# -*- coding: utf-8 -*- noqa
"""
Created on Fri Dec 27 19:31:14 2024

@author: Sergio
"""
import os
import re
import sys

from typing import Tuple

import numpy as np
import pandas as pd

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
        echo(f'Files to read: {files}')

    separadores = r'[._]'
    unique_recordings = 0
    recording_count = 0    
    patient_start_win = [0]
    recording_start_win = [0]
    
    for i in range(0, len(files), 2):
        if DEBUG:
            echo('-' * 10)
            echo(f'Reading: "{files[i]}" and "{files[i+1]}"')

        patient_df = pd.read_parquet(
            os.path.join(path_root_directory, files[i+1]),
        )
        
        patient_data_np = np.load(
            os.path.join(path_root_directory, files[i]),
            allow_pickle=True,
        )

        filenames_array = patient_df['filename'].to_numpy()
        unique_recordings += len(np.unique(filenames_array))

        patient_windows = patient_data_np['EEG_win'].astype(np.float32)

        patient_classes = patient_df['class'].to_numpy(dtype=np.int64)
        
        if 'windows' not in locals():     # Primera iteración
            windows = patient_windows
            classes = patient_classes

        else:
            patient_start_win.append(len(windows))
            windows = np.vstack((windows, patient_windows))
            classes = np.hstack((classes, patient_classes))
            
        for i in range(filenames_array.shape[0]):
            new_record = re.split(separadores, filenames_array[i])[1]
            
            if 'last_record' not in locals():
                pass
            
            elif new_record != last_record:
                recording_start_win.append(recording_count)
                
            recording_count += 1
            last_record = new_record
        
        if DEBUG:
            echo(f'Windows shape: {windows.shape}')
            echo(f'Classes shape: {classes.shape}')
            echo(f'Patients starting position lenght: {len(patient_start_win)}')
            echo(f'Recodings starting position length: {len(recording_start_win)}')

        assert (  
                windows.shape[0] == classes.shape[0] 
        ), 'Loaded data arrays have not the same 0th dimension shape.'
        
    patient_start_win = np.array(patient_start_win)  # Esto es un array de ints
    recording_start_win = np.array(recording_start_win)  # Esto es un array de ints

    assert (
            windows.shape[0] == classes.shape[0]
    ), 'Loaded data arrays have not the same 0th dimension shape.'
    
    assert (
            len(recording_start_win) == unique_recordings
    ),  'The number of unique recordings registered does not correspond to the real number of unique recordings.'

    assert (
            np.all(np.isin(patient_start_win, recording_start_win))
    ),  'The starting positions of the recordings and the patients do not correspond.'
    
    return windows, classes, patient_start_win, recording_start_win


if __name__ == '__main__':
    windows, classes, patient_start_win, recording_start_win = load_seizures(DATA_PATH)
    if DEBUG:
        echo('Loaded:')
        echo(windows.shape, str(sys.getsizeof(windows)))
        echo(classes.shape, str(sys.getsizeof(classes)))
        echo(patient_start_win.shape, str(sys.getsizeof(patient_start_win)))
        echo(recording_start_win.shape, str(sys.getsizeof(recording_start_win)))