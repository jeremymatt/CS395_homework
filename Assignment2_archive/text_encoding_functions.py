# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 19:32:27 2020

@author: jmatt
"""

import numpy as np

def load_text(filename):
    """
    Loads a text file into memory and returns a list of all the characters in 
    the text file in the order they appear

    Parameters
    ----------
    filename : TYPE string
        The path and filename of the file to be opened

    Returns
    -------
    raw_text : TYPE list
        List of all the characters in the text file in order

    """
    with open(filename) as file:
        #Init an empty list to hold the characters
        raw_text = []
        for line in file:
            #Tack the characters from the current line onto the end of the list
            raw_text.extend(line)
    
    return raw_text


def char_to_onehot(chars, num_chars=256):
    """
    Converts a list of characters to a one-hot array

    Parameters
    ----------
    chars : TYPE list
        A list where each element is a single character
    num_chars : TYPE integer, optional
        The number of characters (classes) in the one_hot encoding. 
        The default is 256.

    Returns
    -------
    one_hot : TYPE numpy array
        The one-hot representation of the input character list

    """
    # Convert to numeric values, keeping only the first 0-255 characters
    ascii_nums = [ord(c) for c in chars if ord(c) < 256]
    # Convert to a 1-hot array
    one_hot = np.eye(num_chars)[ascii_nums]
    
    return one_hot


def onehot_to_char(one_hot):
    """
    Converts a one-hot array to a list of characters

    Parameters
    ----------
    one_hot : TYPE numpy array
        One-hot representation of a list of characters

    Returns
    -------
    text : TYPE list
        A list where each element in the list is a single character

    """
    #For each row in the one-hot array, find the index of the 1 and then 
    #convert to an ascii character
    text = [chr(np.where(r==1)[0][0]) for r in one_hot]
    
    return text

def sample_one_hot():
    """
    Generates a sample one-hot array for testing purposes

    Returns
    -------
    one_hot : TYPE numpy array
        a one-hot array

    """
    one_hot = np.array([[0,0,0,1],
                        [0,0,0,1],
                        [0,0,1,0],
                        [1,0,0,0],
                        [0,0,0,1],
                        [0,1,0,0],
                        [1,0,0,0],
                        [0,1,0,0],
                        [0,0,0,1],
                        [0,0,0,1],
                        ])
    return one_hot
    


def make_samples(one_hot,time_steps,num_predict):
    """
    

    Parameters
    ----------
    one_hot : TYPE numpy array
        A numpy array of one-hot encodings
    time_steps : TYPE integer
        The number of timesteps (characters) to include in each sample
    num_predict : TYPE integer
        

    Returns
    -------
    samples : TYPE numpy array
        Numpy array of samples (x values) 
        of shape (num_samples,time_steps,features), where features
        is the number of one-hot classes
    predictions : TYPE numpy array
        Numpy array of predictions (y values) 
        of shape (num_samples,time_steps,features), where features
        is the number of one-hot classes

    """
    #Determine the number of characters and the number of classes in the 
    #one-hot array
    num_chars,features = one_hot.shape
    #Determine the number of sample/prediction combinations that can be 
    #generated from the input text
    num_samples = num_chars-time_steps-num_predict+1
    
    """
    num_samples - the number of samples to be created
    Time_steps - The number of timesteps to include in one sample
    Features - the one-hot encoding columns
    """
    
    #Init empty numpy array to hold the samples
    samples = np.zeros((num_samples,time_steps,features))
    #Init an empty numpy array to hold the predictions.  Each prediction is a
    #single vector holding one or more one-hot 
    predictions = np.zeros((num_samples,num_predict*features))
    
    #Loop through generating each sample/prediction set
    for i in range(num_samples):
        #Index of the start of the sample in the one-hot array
        samp_start = i
        #Index of the end of the sample in the one-hot array
        samp_end = i+time_steps
        #Index of the end of the prediction in the one-hot array
        pred_end = i+time_steps+num_predict
        #Extract the sample from the one-hot array
        sample = one_hot[samp_start:samp_end,:]
        #Extract the prediction from the one-hot array
        prediction = one_hot[samp_end:pred_end,:]
        #Store the sample in it's array
        samples[i,:,:] = sample
        predictions[i,:] = prediction.flatten()
        
    return (samples,predictions)
    
    
    
    
def print_text(one_hot):
    """
    Prints one-hot characters to the screen

    Parameters
    ----------
    one_hot : TYPE numpy array
        Characters encoded in a one-hot array

    Returns
    -------
    None.

    """
    #Convert from one-hot to list of characters
    chars = onehot_to_char(one_hot)
    #Join and print the characters
    print(''.join(chars))
    
    
def generate_text(trained_model,num_to_generate,seed):
    """
    Uses a trained model to generate character predictions from an input seed

    Parameters
    ----------
    trained_model : TYPE keras serial model
        The model with trained weights and an input shape of one sample
    num_to_generate : TYPE integer
        The number of characters to generate
    seed : TYPE numpy array
        A one-hot array of characters to use as a seed

    Returns
    -------
    chars : TYPE
        DESCRIPTION.

    """
    #Determine the time steps (sequence length) of the sample and the number
    #of classes in the one-hot encoding
    time_steps,num_chars = seed.shape
    
    #Generate an array of zeros to hold the generated characters
    chars = np.zeros((num_to_generate,num_chars))
    #Set the first column equal to one (so each row is a valid one-hot vector)
    chars[:,0] = 1
    
    #for each character to generate
    for i in range(num_to_generate):
        #Reshape the seed to be three dimensional with dimensions:
        #(one-sample, sequence-length, number-of-one-hot-classes)
        output = trained_model.predict(seed.reshape((1,time_steps,num_chars)))
        #Convert the output class probabilities to a one-hot vector (predict
        #the max value to be the class)
        new_char = (output == output.max()).astype(int)
        
        #Put the new character vector in the array of generated characters
        chars[i] = new_char
        
        #Shift the seed one to the "left" (drop the first character)
        seed[:-1] = seed[1:]    
        #Insert the new character at the end (right side) of the string
        seed[-1] = new_char
        
    return chars
    
    
    