# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 19:32:27 2020

@author: jmatt
"""


def load_text(filename):
    with open(filename) as file:
        # raw_text = file.read()
        raw_text = []
        for line in file:
            raw_text.extend(line)
        
    #remove excess white space
    # words = raw_text.split()
    clean_text = raw_text
    
    return clean_text