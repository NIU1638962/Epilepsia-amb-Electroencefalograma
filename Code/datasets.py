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
        "__is_lstm",
        "__is_personalized",
        "__jump_amount",
        "__jump_index",
        "__len",
        "__len_patients",
        "__len_recordings",
        "__len_windows",
        "__path_root_directory",
        "__patient",
        "__patient_start_idexes",
        "__recordings_start_idexes",
        "__windows",
    )

###############################################################################
#                             Overloaded Operators                            #

    def __init__(self, path_root_directory: str):
        self.__path_root_directory = path_root_directory

        self.__classes = Tensor()
        self.__patient_start_idexes = np.ndarray()
        self.__recordings_start_idexes = np.ndarray()
        self.__windows = Tensor()

        self.__len = 0
        self.__len_patients = 0
        self.__len_recordings = 0
        self.__len_windows = 0

        self.__patient = None
        self.__is_lstm = False
        self.__is_personalized = False

        self.__load_data()
        self.__calculate_internal_indexes()

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

        for i, e in enumerate(index):
            if e >= self.__jump_index:
                index[i] = e + self.__jump_amount

        return self.__windows[index], self.__classes[index]

    def __len__(self) -> int:
        """
        Length of the Dataset.

        Returns
        -------
        integer
            Length of windows or recordings dependin on is_lstm.

        """
        return self.__len

###############################################################################


###############################################################################
#                              Protected Methods                              #

    def __load_data(self):  # noqa
        windows, classes, patient_start_idexes, recordings_start_idexes = load_seizures(
            self.__path_root_directory
        )

        self.__windows = from_numpy(windows)
        self.__classes = from_numpy(classes)
        self.__patient_start_idexes = patient_start_idexes
        self.__recordings_start_idexes = recordings_start_idexes

        self.__len_windows = len(self.__windows)
        self.__len_patients = len(self.__patient_start_idexes)
        self.__len_recordings = len(self.__recordings_start_idexes)

    def __calculate_internal_indexes(self):
        if self.__is_lstm:
            if self.__patient is None:
                self.__jump_index = self.__len_patients
                end_index = self.__len_patients

            else:
                self.__jump_index = np.where(
                    (
                        self.__recordings_start_idexes
                    ) == (
                        self.__patient_start_idexes[self.__patient]
                    )
                )

                if self.__patient == self.__len_patients - 1:
                    end_index = self.__len_recordings - 1

                else:
                    end_index = np.where(
                        (
                            self.__recordings_start_idexes
                        ) == (
                            self.__patient_start_idexes[self.__patient + 1]
                        )
                    )

            self.__jump_amount = end_index - self.__jump_index
            if self.__is_personalized:
                self.__len = self.__jump_amount
                self.__jump_amount = self.__recordings_start_idexes[
                    self.__jump_index
                ]
                self.__jump_index = 0

            else:
                self.__len = self.__len_recordings - self.__jump_amount
                self.__jump_index = self.__recordings_start_idexes[
                    self.__jump_index
                ]
                end_index = self.__recordings_start_idexes[end_index]
                self.__jump_amount = end_index - self.__jump_index

        else:
            if self.__patient is None:
                self.__jump_index = self.__len_windows
                end_index = self.__len_windows

            else:
                self.__jump_index = self.__patient_start_idexes[self.__patient]

                if self.__patient == self.__len_patients - 1:
                    end_index = self.__len_windows - 1

                else:
                    end_index = self.__patient_start_idexes[self.__patient + 1]

            self.__jump_amount = end_index - self.__jump_index
            if self.__is_personalized:
                self.__len = self.__jump_amount
                self.__jump_amount = self.__jump_index
                self.__jump_index = 0

            else:
                self.__len = self.__len_windows - self.__jump_amount

###############################################################################


###############################################################################
#                                  Properties                                 #

    @property  # noqa
    def patient(self) -> int:
        """
        Retrive Int with the patient being left out of train set.

        Returns
        -------
        Int
            Int indicating which patient is being left out in the actual fold.

        """
        return self.__patient

    @patient.setter
    def patient(self, new_patient: int):
        """
        Set new patient to select of train set.

        Parameters
        ----------
        new_patient : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.__patient = new_patient
        self.__calculate_internal_indexes()

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
    def is_lstm(self, new_is_lstm: bool):
        """
        Set new lstm training state.

        Parameters
        ----------
        new_is_lstm : TYPE
            state of a lstm model being trained.

        Returns
        -------
        None.

        """
        self.__is_lstm = new_is_lstm
        self.__calculate_internal_indexes()

    @property
    def is_personalized(self) -> bool:
        """
        Return if the model is in personalized mode instead of generalized.

        Returns
        -------
        bool
            DESCRIPTION.

        """
        return self.__is_personalized

    @is_personalized.setter
    def is_personalized(self, new_is_personalized: int):
        self.__is_personalized = new_is_personalized
        self.__calculate_internal_indexes()


###############################################################################
if __name__ == "__main__":
    dataset = SeizuresDataset(DATA_PATH)
    if DEBUG:
        echo(dataset.windows.shape)
        echo(dataset.classes.shape)
        echo(dataset.patients_ids.shape)
        echo(dataset.recordings.shape)
