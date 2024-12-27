import os
import numpy as np
import pandas as pd
import re


def load_seizures(path_root_directory: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Args:
        path_root_directory (str): path to dataset

    Returns:
        tuple[all_windows, all_classes, all_patient_ids, all_recordings]: 
        
        all_windows: todas las windows existentes concatenadas en orden de paciente
        all_classes: la clase del window correspondiente
        all_patient_ids: indica la pertenencia de cada window a un paciente
        all_recordings: indica la pertenencia de cada window a un recording relativo a un paciente
    """
    
    files = os.listdir(path_root_directory)
    separadores = r"[._]"
    all_recordings = []
    for i in range(0,len(files),2):
        patient_df = pd.read_parquet(path_root_directory+files[i+1])
        patient_data_np = np.load(path_root_directory+files[i], allow_pickle=True)
        
        patient_id = files[i].split("_")[0][-2:]
        filenames_array = patient_df["filename"].to_numpy()

        patient_windows = patient_data_np["EEG_win"]
        patient_classes = patient_df["class"].to_numpy()
        patient_id_array = np.full((patient_windows.shape[0],), int(patient_id), dtype=int)
        if "all_windows" not in locals():     # Primera iteración
            all_windows = patient_windows  
            all_classes = patient_classes
            all_patient_ids = patient_id_array

        else:
            all_windows = np.vstack((all_windows, patient_windows))
            print(patient_classes.shape)
            all_classes = np.hstack((all_classes, patient_classes))
            print(all_classes.shape)
            all_patient_ids = np.hstack((all_patient_ids, patient_id_array))

        for i in range(filenames_array.shape[0]):
            all_recordings.append(re.split(separadores, filenames_array[i])[1])
    all_recordings = np.array(all_recordings)
    
    return all_windows, all_classes, all_patient_ids, all_recordings
            



            
            
all_windows, all_classes, all_patient_ids, all_recordings = load_seizures("data/")
print(all_windows.shape)
print(all_classes.shape)
print(all_patient_ids.shape)
print(all_recordings.shape)
