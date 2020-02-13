# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:19:07 2020

@author: jmatt
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('block_tuning.xlsx')

plt.figure()

plt.plot(data['epoch'],data['test'],label='Training')
plt.plot(data['epoch'],data['train'],label='Validation')
plt.ylabel('One Class Accuracy')
plt.xlabel('Epoch')
plt.legend()