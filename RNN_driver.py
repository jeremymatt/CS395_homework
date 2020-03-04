# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:08:53 2020

@author: jmatt
"""


import numpy as np
import os

from keras import Sequential 
from keras.layers import Conv2D, MaxPooling2D,LSTM
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import layers
from keras import optimizers
from keras import applications
from keras import Model

from text_encoding_functions import *

path_base = './data'
# files = [f for f in os.listdir('./data') if ((os.path.isfile(f))&(f.split('.')[-1] == 'txt'))]
# files = [f for f in os.listdir('data\\') if ((os.path.isfile(f))&(True))]

# files = [os.path.join(path_base,f) for f in os.listdir(path_base) if os.path.isfile(os.path.join(path_base,f))]
files = [os.path.join(path_base,f) for f in os.listdir(path_base) if f.split('.')[-1] == 'txt']

time_steps = 50
num_predict = 1
LSTM_hidden_units = 64
batch_size = 128
num_chars = 256
num_units = 256

model = Sequential()
model.add(LSTM(
    units = num_units,
    batch_input_shape = (batch_size,time_steps,num_chars),
    stateful=True,
    return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(num_chars*num_predict,activation='softmax'))

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics = ['accuracy'])

# for file in files:
#     print(file)
#     text = load_text(file)
#     one_hot = char_to_onehot(text)
#     (X,y) = make_samples(one_hot,time_steps,num_predict)
#     useable_samples = int(X.shape[0]/batch_size)*batch_size
#     X = X[:useable_samples]
#     y = y[:useable_samples]

file = files[0]
print(file)
text = load_text(file)
one_hot = char_to_onehot(text)
(X,y) = make_samples(one_hot,time_steps,num_predict)
useable_samples = int(X.shape[0]/batch_size)*batch_size
X = X[:useable_samples]
y = y[:useable_samples]
    

# # define the checkpoint
# filepath=f"weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]

model.fit(X, y, epochs=1, batch_size=batch_size)
    



pred_model = Sequential()
pred_model.add(LSTM(
    units = num_units,
    batch_input_shape = (1,time_steps,num_chars),
    stateful=True,
    return_sequences=False))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(num_chars*num_predict,activation='softmax'))

    
    
    # num_chars = len(text)
    
    # spaces = []
    # for i in range(time_steps):
    #     spaces.append(' ')

    # text.extend(spaces)
