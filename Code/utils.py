# -*- coding: utf-8 -*- noqa
"""
Created on Sat Dec 28 19:00:36 2024

@author: Joel Tapia Salvador
"""
import os


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
