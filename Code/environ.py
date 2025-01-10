# -*- coding: utf-8 -*- noqa
"""
Created on Sat Dec 28 19:13:43 2024

@author: Joel  Tapia Salvador
"""
import os
import sys

from utils import echo

DEBUG = True

PLATFORM = sys.platform.lower()

USER = ''

if PLATFORM == 'linux':
    USER = os.getenv('USER')
elif PLATFORM == 'win32':
    USER = os.getenv('USERNAME')
elif PLATFORM == 'darwin':
    USER = os.getenv('')  # No se el nombre de la variable de environment
    # "user" en mac

USER.lower()

if USER == 'maed02':
    DATA_PATH = '../../../maed/EpilepsyDataSet/'
    RESULTS_PATH = '../Results/'
    TRAINED_MODELS_PATH = '../Trained Models/'
elif USER == 'Usuario':
    DATA_PATH = "../EpilepsyDataSet"
    RESULTS_PATH = '../Results/'
    TRAINED_MODELS_PATH = '../Trained Models/'
else:
    DATA_PATH = '../Data/'
    RESULTS_PATH = '../Results/'
    TRAINED_MODELS_PATH = '../Trained Models/'

if DEBUG:
    echo(f'Platform: "{PLATFORM}"')
    echo(f'User: "{USER}"')
    echo(f'Path to data: "{DATA_PATH}"')
    echo(f'Path to results: "{RESULTS_PATH}"')
    echo(f'Path to trained models: "{TRAINED_MODELS_PATH}"')
