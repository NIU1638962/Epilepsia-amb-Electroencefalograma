# -*- coding: utf-8 -*- noqa
"""
Created on Sat Dec 28 15:44:49 2024

@author: Joel Tapia Salvador
"""
from torch.utils.data import Dataset, DataLoader


def create_dataloader(dataset: Dataset, batch: int, shuffle = True) -> DataLoader:
    """
    Create a dataloader with the given dataset and batch size.

    Parameters
    ----------
    dataset : Torch Dataset
        Dataset that will be loaded.
    batch : integer
        Size of the batch that will be loaded each time.

    Returns
    -------
    Torch DataLoader
        Torch DataLoader that implements the batch itself.

    """
    # if not issubclass(dataset, Dataset):
    #    raise TypeError('"dataset" is not a subclass of Torch Dataset.')

    return DataLoader(dataset, batch_size=batch, shuffle=shuffle)
