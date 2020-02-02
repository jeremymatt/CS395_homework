# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 21:06:39 2020

@author: jmatt
"""




import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator as IDG
from keras import Sequential 
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import layers
from keras import optimizers
from keras.models import load_model




on_windows = False
if on_windows:
    data_directory = 'D:\\Data\\Sketches\\png'
    path_delim = '\\'
else:
    data_directory = '../data/Sketches/png'
    path_delim = '/'

# data_list = 'filelist.txt'

# filename = '{}{}{}'.format(data_directory,path_delim,data_list)

# image_list = pd.read_csv(filename,names=['files'])


# image_list = [(fn.split('/')[0],fn.split('/')[1]) for fn in image_list['files']]


# import matplotlib.image as mpimg

# file_tpl = image_list[100]
# fn = '{}{}{}{}{}'.format(data_directory,path_delim,file_tpl[0],path_delim,file_tpl[1])

# fn = f'{data_directory}{path_delim}{file_tpl[0]}{path_delim}{file_tpl[1]}'

# img=mpimg.imread(fn)
# imgplot = plt.imshow(img)
# plt.show()



# From https://stackoverflow.com/questions/46717742/split-data-directory-into-training-and-test-directory-with-sub-directory-structu
#image dimensions
img_height = 150
img_width = 150


train_datagen = IDG(
    samplewise_std_normalization=True,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    validation_split = 0.2)

batch_size = 32
class_mode = 'categorical'
color_mode = 'grayscale'
shuffle = True

train_generator = train_datagen.flow_from_directory(
    data_directory,
    target_size = (img_width,img_height),
    batch_size = batch_size,
    class_mode = class_mode,
    color_mode = color_mode,
    shuffle = shuffle,
    subset = 'training')

validation_generator = train_datagen.flow_from_directory(
    data_directory,
    target_size = (img_width,img_height),
    batch_size = batch_size,
    class_mode = class_mode,
    color_mode = color_mode,
    shuffle = shuffle,
    subset = 'validation')

test_generator = train_datagen.flow_from_directory(
    data_directory,
    target_size = (img_width,img_height),
    class_mode = None,
    color_mode = color_mode,
    batch_size = 32,
    subset = 'validation')

model = load_model('first_try.h5')


acc = model.evaluate_generator(validation_generator, steps=2,verbose=1)

print(acc)