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
batch_size = 128 #Training batch size
num_chars = 256 #Number of characters (number of one-hot classes)
num_units = 256 #Number of hidden units in the network
num_to_generate = 500 #Number of samples to generate

#Define the training model and add layers with an input shape that includes
#more than one sample
model = Sequential()
model.add(SimpleRNN(
    units = num_units,
    batch_input_shape = (batch_size,time_steps,num_chars),
    stateful=True,
    return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(num_chars*num_predict,activation='softmax'))

#Define a model to use for character generation that takes only one sample
#at a time
trained_model = Sequential()
trained_model.add(SimpleRNN(
    units = num_units,
    batch_input_shape = (1,time_steps,num_chars),
    stateful=True,
    return_sequences=False))
trained_model.add(Dropout(0.2))
trained_model.add(Dense(num_chars*num_predict,activation='softmax'))

#Compile the model
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics = ['accuracy'])

#The VACC won't read the text files I'm using, even when I save a cleaned 
#version that only has the 0-255 ascii characters.  Therefore I saved the 
#one-hot encoding for the training data and load that directly if it exists 
#rather than trying to process the text file.
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
    
#Generate the training samples and associated labels.    
(X,y) = make_samples(one_hot,time_steps,num_predict)
#Drop the partial batch at the end of the X and y
useable_samples = int(X.shape[0]/batch_size)*batch_size
X = X[:useable_samples]
y = y[:useable_samples]
    
#Define a file name to save the results
file = 'RNN_results.txt'
#List of number of epochs to train the network for.
epochs_list = [1,1,1,1,1,5,10,10,10,10,20,20,20,20,20,50,50,50,50,50,50,50,100,100,100,100,100]

#Clear any existing data in the output file
with open(file,'w') as outfile:
    outfile.write('')
    
#Iterate through each training duration
for ind,epochs in enumerate(epochs_list):
    #Fit the model for the current number of epochs
    model.fit(X, y, epochs=epochs, batch_size=batch_size)
        
    #Extract the weights from the training model to set the weights for the
    #character generation model
    weights = model.get_weights()
    trained_model.set_weights(weights)
    
    #Select a sample from X to be the seed
    seed = np.array(X[100])
    
    #Generate the results and output to file
    with open(file,'a') as outfile:
        #Write the header line with number of epochs, seed, etc
        outfile.write('Using seed of length {} generated {} characters after {} epochs.\nSeed: \'{}\'\n'.format(
                    seed.shape[0],
                    num_to_generate,
                    sum(epochs_list[:ind+1]),
                    ''.join(onehot_to_char(seed))))
        #Generate the characters from the current iteration of the trained model
        chars = generate_text(trained_model,num_to_generate,seed)
        #Convert from a one-hot array to a list of characters
        text = onehot_to_char(chars)
        #Join the characters into a string and write to file
        outfile.write('\"{}\"'.format(''.join(text)))
        #Write separator bar
        outfile.write('\n==========================================\n\n')

