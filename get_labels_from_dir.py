# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 18:23:09 2020

@author: jmatt
"""


import pandas as pd
import glob


'''
    For the given path, get the List of all files in the directory tree 
'''
def get_labels_from_dir(
        target_dir,
        path_sep,
        img_types = {'png','jpg','jpeg','PNG','JPG','"JPEG'}):
    """
    Generates a pandas dataframe of labels and relative paths to files
    from a given target directory.  The label of each file is the name of the
    directory containing the file

    Parameters
    ----------
    target_dir : <string>
        The root directory of the dataset
    path_sep : <string>
        The directory separator for the current OS
    img_types : <set>, optional
        Set containing allowable file extensions. 
        The default is {'png','jpg','jpeg','PNG','JPG','"JPEG'}.

    Returns
    -------
    label_df : <pandas dataframe>
        Contains a column of labels and file paths relative to the root dir

    """
    
    #Generate a list of all files in the directory
    raw_files = glob.glob(target_dir + '/**/*.*', recursive=True)
    
    #Init empty lists
    labels = []
    files = []
    
    #Update the path separator for text concatenation if OS is windows
    if path_sep == '\\':
        p_s = '\\\\'
    else:
        p_s = path_sep
        
    #Loop through each of the files
    for f in raw_files:
        #Split on the directory separator
        file_path = f.split(target_dir)[1]
        parts = file_path.split(path_sep)
        if parts[-1].split('.')[1] in img_types:
            labels.append(parts[-2])
            files.append(f'{p_s}{parts[-3]}{p_s}{parts[-2]}{p_s}{parts[-1]}')
            
    label_df = pd.DataFrame({'labels':labels,'files':files})
        
    return label_df
    
    