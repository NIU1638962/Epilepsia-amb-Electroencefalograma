# -*- coding: utf-8 -*- noqa
"""
Created on Sat Dec 28 15:27:34 2024

@author: Joel Tapia Salvador
"""

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
        "__path_root_directory",
        "__patients_ids",
        "__recordings",
        "__windows",
    )

###############################################################################
#                             Overloaded Operators                            #

    def __init__(self, path_root_directory: str):
        self.__path_root_directory = path_root_directory

        self.__classes = Tensor()
        self.__patients_ids = Tensor()
        self.__recordings = Tensor()
        self.__windows = Tensor()

        self.__len = 0

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
        return self.__windows[index], self.__classes[index]

    def __len__(self) -> int:
        """
        Length of the Dataset.

        Returns
        -------
        integer
            Length.

        """
        return self.__len

###############################################################################


###############################################################################
#                              Protected Methods                              #

    def __load_data(self):  # noqa
        windows, classes, patients_ids, recordings = load_seizures(
            self.__path_root_directory
        )

        self.__windows = from_numpy(windows)
        self.__classes = from_numpy(classes)
        self.__patients_ids = from_numpy(patients_ids)
        self.__recordings = recordings

        self.__len = len(self.__windows)

###############################################################################


###############################################################################
#                                  Properties                                 #

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
