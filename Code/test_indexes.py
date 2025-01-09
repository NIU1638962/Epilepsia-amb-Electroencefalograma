# -*- coding: utf-8 -*- noqa
"""
Created on Thu Jan  9 21:37:15 2025

@author: Joel Tapia Salvador
"""
import numpy as np
from torch import from_numpy, Tensor, int32, tensor
from torch.utils.data import Dataset

from dataloaders import create_dataloader


class TestDataset(Dataset):
    """Torch Dataset with the seizures information loaded."""

    __slots__ = (
        "__classes",
        "__is_lstm",
        "__is_personalized",
        "__is_test",
        "__jump_amount",
        "__jump_amount_test",
        "__jump_index",
        "__jump_index_test",
        "__len",
        "__len_patients",
        "__len_recordings",
        "__len_windows",
        "__path_root_directory",
        "__patient",
        "__patient_start_idexes",
        "__recordings_start_idexes",
        "__test_recording",
        "__windows",
    )

###############################################################################
#                             Overloaded Operators                            #

    def __init__(self, path_root_directory: str):
        self.__path_root_directory = path_root_directory

        self.__classes = Tensor()
        self.__patient_start_idexes = np.ndarray([])
        self.__recordings_start_idexes = np.ndarray([])
        self.__windows = Tensor()

        self.__len = 0
        self.__len_patients = 0
        self.__len_recordings = 0
        self.__len_windows = 0

        self.__patient = None
        self.__test_recording = None

        self.__is_lstm = False
        self.__is_personalized = False
        self.__is_test = False

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

        if index >= self.__jump_index:
            index += self.__jump_amount

        if index >= self.__jump_index_test:
            index += self.__jump_amount_test

        if self.__is_lstm:
            start = self.__recordings_start_idexes[index]
            if index + 1 >= self.__len_recordings:
                end = self.__len_windows
            else:
                end = self.__recordings_start_idexes[index + 1]
            index = tensor(range(start, end), dtype=int32)

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
        windows = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 2],
                [0, 1, 3],
                [0, 1, 4],
                [0, 1, 5],
                [0, 1, 6],
                [0, 2, 7],
                [0, 2, 8],
                [0, 2, 9],
                [0, 3, 10],
                [0, 3, 11],
                [0, 3, 12],
                [1, 4, 13],
                [1, 4, 14],
                [1, 4, 15],
                [1, 4, 16],
                [1, 5, 17],
                [1, 5, 18],
                [1, 5, 19],
                [1, 6, 20],
                [1, 6, 21],
                [1, 6, 22],
                [2, 7, 23],
                [2, 7, 24],
                [2, 7, 25],
                [2, 8, 26],
                [2, 8, 27],
                [2, 8, 28],
                [2, 9, 29],
            ],
        )

        classes = np.array([int(bool(i % 10))for i in range(30)])

        patient_start_idexes = np.array(
            [
                0,
                13,
                23,
            ],
        )

        recordings_start_idexes = np.array(
            [
                0,
                3,
                7,
                10,
                13,
                17,
                20,
                23,
                26,
                29,
            ],
        )

        self.__windows = from_numpy(windows)
        self.__classes = from_numpy(classes)
        self.__patient_start_idexes = patient_start_idexes
        self.__recordings_start_idexes = recordings_start_idexes

        self.__len_windows = len(self.__windows)
        self.__len_patients = len(self.__patient_start_idexes)
        self.__len_recordings = len(self.__recordings_start_idexes)
        self.__len_patient_recordings = None

        assert (
            self.__len_windows == 30
        ), (
            f'Length windows: {self.__len_windows} != 30'
        )

        assert (
            self.__len_patients == 3
        ), (
            f'Length patients: {self.__len_patients} != 3'
        )

        assert (
            self.__len_recordings == 10
        ), (
            f'Length recordings: {self.__len_recordings} != 10'
        )

    def __calculate_internal_indexes(self):
        if self.__patient is None:
            start_recordings_index = self.__len_recordings
            end_recordings_index = self.__len_recordings

        else:
            start_recordings_index = int(np.where(
                (
                    self.__recordings_start_idexes
                ) == (
                    self.__patient_start_idexes[self.__patient]
                )
            )[0][0])

            if self.__patient == self.__len_patients - 1:
                end_recordings_index = self.__len_recordings

            else:
                end_recordings_index = int(np.where(
                    (
                        self.__recordings_start_idexes
                    ) == (
                        self.__patient_start_idexes[self.__patient + 1]
                    )
                )[0][0])

        self.__len_patient_recordings = (
            end_recordings_index - start_recordings_index
        )

        if self.__test_recording is None:
            start_test_recordings_index = self.__len_recordings - 1
            end_test_recordings_index = self.__len_recordings - 1
        else:
            start_test_recordings_index = (
                self.__test_recording + start_recordings_index
            )

            end_test_recordings_index = start_test_recordings_index + 1

        if self.__is_lstm:
            total_len_index = self.__len_recordings
            if self.__patient is None:
                start_index = total_len_index
                end_index = total_len_index
                start_test_index = total_len_index
                end_test_index = total_len_index

            else:
                start_index = start_recordings_index
                end_index = end_recordings_index
                start_test_index = start_test_recordings_index
                end_test_index = end_test_recordings_index

        else:
            total_len_index = self.__len_windows
            if self.__patient is None:
                start_index = total_len_index
                end_index = total_len_index
                start_test_index = total_len_index
                end_test_index = total_len_index

            else:
                start_index = self.__patient_start_idexes[self.__patient]

                if self.__patient == self.__len_patients - 1:
                    end_index = self.__len_windows

                else:
                    end_index = self.__patient_start_idexes[self.__patient + 1]

                start_test_index = self.__recordings_start_idexes[
                    start_test_recordings_index
                ]

                if end_test_recordings_index >= self.__len_recordings:
                    end_test_index = total_len_index
                else:
                    end_test_index = self.__recordings_start_idexes[
                        end_test_recordings_index
                    ]

        len_index = end_index - start_index

        if self.__is_lstm:
            pass
        else:
            pass

        if self.__is_personalized:
            self.__len = len_index
            self.__jump_amount = start_index
            self.__jump_index = 0
        else:
            if self.__is_test:
                self.__len = len_index
                self.__jump_amount = start_index
                self.__jump_index = 0

                self.__jump_amount_test = 0
                self.__jump_index_test = 0

            else:
                self.__len = total_len_index - len_index
                self.__jump_amount = len_index
                self.__jump_index = start_index

                self.__jump_amount_test = 0
                self.__jump_index_test = 0

        # #######################################################################

        # if self.__is_lstm:
        #     if self.__patient is None:
        #         pass

        #     else:

        #         if self.__patient == self.__len_patients - 1:
        #             end_index = self.__len_recordings - 1

        #         else:
        #             end_index = np.where(
        #                 (
        #                     self.__recordings_start_idexes
        #                 ) == (
        #                     self.__patient_start_idexes[self.__patient + 1]
        #                 )
        #             )

        #     self.__jump_amount = end_index - self.__jump_index
        #     if (
        #             self.__is_personalized
        #     ) or (
        #             (
        #                 not self.__is_personalized
        #             ) and (
        #                 self.__is_test
        #             )
        #     ):
        #         self.__len = self.__jump_amount
        #         self.__jump_amount = self.__jump_index
        #         self.__jump_index = 0

        #     else:
        #         self.__len = self.__len_recordings - self.__jump_amount
        #         self.__jump_index = self.__jump_index

        # else:
        #     if self.__patient is None:
        #         self.__jump_index = self.__len_windows
        #         end_index = self.__len_windows
        #         self.__len_patient_recordings = None

        #     else:
        #         self.__jump_index = self.__patient_start_idexes[self.__patient]

        #         if self.__patient == self.__len_patients - 1:
        #             end_index = self.__len_windows - 1

        #         else:
        #             end_index = self.__patient_start_idexes[self.__patient + 1]

        #     self.__jump_amount = end_index - self.__jump_index

        #     if self.__test_recording is None:
        #         self.__jump_index_test = self.__len_windows
        #         self.__jump_amount_test = 0
        #     # else:

        #     if (
        #             self.__is_personalized
        #     ) or (
        #             (
        #                 not self.__is_personalized
        #             ) and (
        #                 self.__is_test
        #             )
        #     ):
        #         self.__len = self.__jump_amount
        #         self.__jump_amount = self.__jump_index
        #         self.__jump_index = 0

        #     else:
        #         self.__len = self.__len_windows - self.__jump_amount

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

    @property
    def is_test(self) -> bool:
        """
        Return if the model is in test mode or not (if not is train mode).

        Returns
        -------
        bool
            DESCRIPTION.

        """
        return self.__is_test

    @is_test.setter
    def is_test(self, new_is_test: bool):
        self.__is_test = new_is_test
        self.__calculate_internal_indexes()

    @property  # noqa
    def len_patient_recordings(self) -> int:
        """
        Retrive Int number of recordings for actual patient.

        Returns
        -------
        Int
            Int indicating number of recordings of the actual patient.

        """
        return self.__len_patient_recordings


###############################################################################
if __name__ == "__main__":
    dataset = TestDataset('')

    loader = create_dataloader(dataset, 2, False)

    for data in loader:
        print(data)

    print('-'*10)

    dataset.is_lstm = True

    loader = create_dataloader(dataset, 1, False)

    for data in loader:
        print(data)

    for i in range(3):
        print('-' * 10)
        print(i)
        dataset.patient = i

        dataset.is_test = False

        dataset.is_lstm = False

        print('='*10)

        loader = create_dataloader(dataset, 2, False)

        for data in loader:
            print(data)

        print('='*10)

        dataset.is_test = True

        loader = create_dataloader(dataset, 2, False)

        for data in loader:
            print(data)

        print('='*10)

        dataset.is_test = False

        dataset.is_lstm = True

        loader = create_dataloader(dataset, 1, False)

        for data in loader:
            print(data)

        print('='*10)

        dataset.is_test = True

        loader = create_dataloader(dataset, 1, False)

        for data in loader:
            print(data)
