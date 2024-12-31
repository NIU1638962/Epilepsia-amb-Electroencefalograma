# -*- coding: utf-8 -*- noqa
"""
Created on Sat Dec 28 15:27:34 2024

@author: Joel Tapia Salvador
"""

import numpy as np
from torch import from_numpy, Tensor
from torch.utils.data import Dataset

from load_datasets import load_seizures

from environ import DATA_PATH, DEBUG
from utils import echo


class SeizuresDataset(Dataset):
    """Torch Dataset with the seizures information loaded."""

    __slots__ = (
        "__classes",
        "__len",
        "__len_pat",
        "__len_rec",
        "__path_root_directory",
        "__patient_start_idx",
        "__recordings_start_idx",
        "__windows",
        "__patient_out",
        "__is_lstm"
    )

###############################################################################
#                             Overloaded Operators                            #

    def __init__(self, path_root_directory: str):
        self.__path_root_directory = path_root_directory

        self.__classes = Tensor()
        self.__patient_start_idx = np.ndarray()
        self.__recordings_start_idx = np.ndarray()
        self.__windows = Tensor()

        self.__len = 0
        self.__len_pat = 0
        self.__len_rec = 0
        self.__patient_out = None
        self.__is_lstm = False
        self.__load_data()

    def __getitem__(self, index: Tensor) -> Tensor:
        """
        Return a slice of the dataset.

        Parameters
        ----------
        index : Torch Tensor
            Indexes to return.

        Returns
        -------
        Tensor
            Slice of the dataset.

        """
        # cuando sea lstm entrara un tensor con los indices de recordings que quiere devolver.
        # la logica externa contara con una longitud del vector de recordings menor a la real.
        # por lo tanto tener en cuenta el paciente left out y los indices mayor a este extenderlos
        # en la medida de recordings de este paciente para coger los windows correspondientes.

        if self.__is_lstm:
            pass

        return self.__windows[index], self.__classes[index]

    def __len__(self) -> int:
        """
        Length of the Dataset.

        Returns
        -------
        integer
            Length of windows or recordings dependin on is_lstm.

        """
        if self.__is_lstm:
            if self.__patient_out is None:
                return self.__len_rec
            # Calculate the number of recordings of the patient of test and
            # we substract it from the total number of recordings
            idx1 = np.where(self.__recordings_start_idx ==
                            self.__patient_start_idx[self.__patient_out])

            if self.__patient_out == self.__len_pat - 1:
                idx2 = self.__len_rec - 1

            else:
                idx2 = np.where(self.__recordings_start_idx ==
                                self.__patient_start_idx[self.__patient_out+1])

            if self.__patient_out is None:
                return self.__len

            patient_out_recordings = idx2 - idx1
            return self.len__rec - patient_out_recordings

        # Calculate the number of windows of the patient of test and
        # we substract it from the total number of windows
        idx1 = self.__patient_start_idx[self.__patient_out]

        if self.__patient_out == self.__len_pat - 1:
            idx2 = self.__len - 1

        else:
            idx2 = self.__patient_start_idx[self.__patient_out+1]

        patient_out_wins = idx2 - idx1
        return self.__len - patient_out_wins

###############################################################################


###############################################################################
#                              Protected Methods                              #


    def __load_data(self):  # noqa
        windows, classes, patient_start_idx, recordings_start_idx = load_seizures(
            self.__path_root_directory
        )

        self.__windows = from_numpy(windows)
        self.__classes = from_numpy(classes)
        self.__patient_start_idx = patient_start_idx
        self.__recordings_start_idx = recordings_start_idx

        self.__len = len(self.__windows)
        self.__len_pat = len(self.__patient_start_idx)
        self.__len_rec = len(self.__recordings_start_idx)

###############################################################################


###############################################################################
#                                  Properties                                 #

    @property  # noqa
    def patient_out(self) -> int:
        """
        Retrive Int with the patient being left out of train set.

        Returns
        -------
        Int
            Int indicating which patient is being left out in the actual fold.

        """
        return self.__patient_out

    @patient_out.setter
    def patient_out(self, new_patient):
        """
        Set new patient to be left out of train set

        Args:
            new_patient (_type_): _description_
        """
        self.__patient_out = new_patient

    @property  # noqa
    def is_lstm(self) -> bool:
        """
        Retrive Bool True if training lstm.

        Returns
        -------
        Bool
            Bool indicating if a lstm model is being trained.

        """
        return self.__is_lstm

    @is_lstm.setter
    def is_lstm(self, new_is_lstm):
        """
        Set new lstm training state

        Args:
        -------
        bool
            new_is_lstm: state of a lstm model being trained
        """
        self.__is_lstm = new_is_lstm

    @property  # noqa
    def classes(self) -> Tensor:
        """
        Retrive Torch Tensor with the classes of the windows.

        Returns
        -------
        Torch Tensor
            Classes of Windows. Is a mask 1 on 1 to the "windows" tensor.

        """
        return self.__classes

    @property
    def patients_ids(self) -> Tensor:
        """
        Retrive Torch Tensor with the Patient's IDs.

        Returns
        -------
        Torch Tensor
            Patient's IDs of Windows. Is a mask 1 on 1 to the "windows" tensor.

        """
        return self.__patients_ids

    @property
    def recordings(self) -> Tensor:
        """
        Retrive Torch Tensor with the Recordings of Patients of the Windows.

        Returns
        -------
        Torch Tensor
            Recordings of Patients of Windows. Is a mask 1 on 1 to the
            "windows" tensor.

        """
        return self.__recordings

    @property
    def windows(self) -> Tensor:
        """
        Retrive Tensor with all the Windows with the data of seizures.

        Returns
        -------
        Tensor
            Windows of seizures.

        """
        return self.__windows

###############################################################################


if __name__ == "__main__":
    dataset = SeizuresDataset(DATA_PATH)
    if DEBUG:
        echo(dataset.windows.shape)
        echo(dataset.classes.shape)
        echo(dataset.patients_ids.shape)
        echo(dataset.recordings.shape)
