# -*- coding: utf-8 -*- noqa
"""
Created on Sat Dec 28 19:00:36 2024

@author: Joel Tapia Salvador
"""
import os

import matplotlib.pyplot as plt


def echo(out: str = "", *outs: str, **kwargs):
    """
    Print to console in realtime.

    Parameters
    ----------
    sep : string, optional
        String inserted between values. The default is " ".
    end : strings
        String appended after the last value. The default is newline.

    Raises
    ------
    TypeError
        Arguments badly given.

    Returns
    -------
    None.

    """
    out = str(out)

    try:
        outs = " ".join(outs)

        if outs != "":
            out = out + " " + outs

    except TypeError as error:
        raise TypeError("One or more of arguments is not a string.") from error

    os.system(f"echo '{out}'")


def plot_multiple_losses(loss_logs: list, path: str, title: str = ''):
    """
    Plot multiple loss curves.

    Parameters
    ----------
    loss_logs : list
        List of dictionaries with the keys "name", "train" and "total_time".
    path : string
        Path to where safe the plot.
    title : str, optional
        Title to append to the title. The default is ''.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(10, 7))
    for i, log in enumerate(loss_logs, start=1):
        name = log.get('name', f'Model {i}')
        total_time = log.get('total_time', 'Unkown')
        losses = log.get('train', [])
        plt.plot(
            range(1, len(losses) + 1),
            losses,
            marker='o',
            label=f'{name} ({total_time}s)',
        )

    # Information
    plt.title(f'Training Loss {title}', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)

    # Save plot
    plt.savefig(path)
