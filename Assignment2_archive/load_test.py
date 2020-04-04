# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:42:26 2020

@author: jmatt
"""


import numpy as np
import os

import copy

from keras import Sequential 
from keras.layers import Conv2D, MaxPooling2D,LSTM,SimpleRNN
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import layers
from keras import optimizers
from keras import applications
from keras import Model
import pickle

from text_encoding_functions import *

path_base = './data'
# files = [f for f in os.listdir('./data') if ((os.path.isfile(f))&(f.split('.')[-1] == 'txt'))]
# files = [f for f in os.listdir('data\\') if ((os.path.isfile(f))&(True))]

# files = [os.path.join(path_base,f) for f in os.listdir(path_base) if os.path.isfile(os.path.join(path_base,f))]
files = [os.path.join(path_base,f) for f in os.listdir(path_base) if f.split('.')[-1] == 'txt']


file = files[0]
print(file)
text = load_text(file)
one_hot = char_to_onehot(text)
    
print(''.join(text))