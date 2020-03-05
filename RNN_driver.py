# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:08:53 2020

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

#Find all the training files in the data directory
path_base = './data'
files = [os.path.join(path_base,f) for f in os.listdir(path_base) if f.split('.')[-1] == 'txt']

#Control constants
time_steps = 50 #length of character sequence in one sample
num_predict = 1 #How many output characters to predict
# LSTM_hidden_units = 64 #Number of hidden units in the network
batch_size = 128 #Training batch size
num_chars = 256 #Number of characters (number of one-hot classes)
num_units = 256 #Number of hidden units in the network
num_to_generate = 500 #Number of samples to generate

model = Sequential()
model.add(SimpleRNN(
    units = num_units,
    batch_input_shape = (batch_size,time_steps,num_chars),
    stateful=True,
    return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(num_chars*num_predict,activation='softmax'))



trained_model = Sequential()
trained_model.add(SimpleRNN(
    units = num_units,
    batch_input_shape = (1,time_steps,num_chars),
    stateful=True,
    return_sequences=False))
trained_model.add(Dropout(0.2))
trained_model.add(Dense(num_chars*num_predict,activation='softmax'))



model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics = ['accuracy'])


try:
    print('loading from pickle')
    with open('one_hot.oh','rb') as f:
        one_hot = pickle.load(f)
except:
    print('Processing file for the first time')
    file = files[0]
    print(file)
    text = load_text(file)
    one_hot = char_to_onehot(text)
    with open('one_hot.oh','wb') as f:
        pickle.dump(one_hot,f)
        
(X,y) = make_samples(one_hot,time_steps,num_predict)
useable_samples = int(X.shape[0]/batch_size)*batch_size
X = X[:useable_samples]
y = y[:useable_samples]
    

file = 'RNN_results.txt'
epochs_list = [1,1,1,1,1,5,10,10,10,10,20,20,20,20,20,50,50,50,50,50,50,50,100,100,100,100,100]
# epochs_list = [1,2]


with open(file,'w') as outfile:
    for ind,epochs in enumerate(epochs_list):
        model.fit(X, y, epochs=epochs, batch_size=batch_size)
            
        
        weights = model.get_weights()
        
        trained_model.set_weights(weights)
        
        seed = np.array(X[100])
        
        seed_str = ''.join(onehot_to_char(seed))
        
        outfile.write('Using seed of length {} generated {} characters after {} epochs.\nSeed: \'{}\'\n'.format(
                    seed.shape[0],
                    num_to_generate,
                    sum(epochs_list[:ind+1]),
                    ''.join(onehot_to_char(seed))))
        chars = generate_text(trained_model,num_to_generate,seed)
        text = onehot_to_char(chars)
        outfile.write('\"{}\"'.format(''.join(text)))
        outfile.write('\n==========================================\n\n')

