# -*- coding: utf-8 -*- noqa
import os
import numpy as np
import pandas as pd
import re


def load_seizures(
        path_root_directory: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    separadores = r"[._]"
    recordings = []
    for i in range(0, len(files), 2):
        patient_df = pd.read_parquet(path_root_directory+files[i+1])
        patient_data_np = np.load(
            path_root_directory+files[i], allow_pickle=True)

        patient_id = files[i].split("_")[0][-2:]
        filenames_array = patient_df["filename"].to_numpy()

        patient_windows = patient_data_np["EEG_win"]
        patient_classes = patient_df["class"].to_numpy()
        patient_id_array = np.full(
            (patient_windows.shape[0],), int(patient_id), dtype=int)
        if "windows" not in locals():     # Primera iteración
            windows = patient_windows
            classes = patient_classes
            patients_ids = patient_id_array

        else:
            windows = np.vstack((windows, patient_windows))
            classes = np.hstack((classes, patient_classes))
            patients_ids = np.hstack((patients_ids, patient_id_array))

        for i in range(filenames_array.shape[0]):
            recordings.append(re.split(separadores, filenames_array[i])[1])
    recordings = np.array(recordings)

    assert (
        (
            windows.shape[0] == classes.shape[0]
        ) and (
            windows.shape[0] == patients_ids.shape[0]
        ) and (
            windows.shape[0] == recordings.shape[0]
        )
    ), "Loaded data arrays have not the same 0th dimension shape."

    return windows, classes, patients_ids, recordings


if __name__ == "__main":
    windows, classes, patients_ids, recordings = load_seizures(
        "../Data/")
    print(windows.shape)
    print(classes.shape)
    print(patients_ids.shape)
    print(recordings.shape)
