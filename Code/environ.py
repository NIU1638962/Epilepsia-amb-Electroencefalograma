# -*- coding: utf-8 -*- noqa
"""
Created on Sat Dec 28 19:13:43 2024

@author: Joel  Tapia Salvador
"""
import os
import sys

platform = sys.platform.lower()

user = None

if platform == 'linux':
    user = os.getenv('USER')
elif platform == 'win32':
    user = os.getenv('USERNAME')

if user == 'maed02':
    DATA_PATH = '../../../maed/EpilepsyDataSet/'
else:
    DATA_PATH = '../Data/'
