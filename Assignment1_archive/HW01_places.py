# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:34:10 2020

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
from get_labels_from_dir import get_labels_from_dir as get_labels



on_windows = True
if on_windows:
    data_directory = 'D:\\Data\\Test'
    path_delim = '\\'
else:
    data_directory = '../data/Places'
    path_delim = '/'


label_df = get_labels(data_directory,path_delim)

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

train_generator = train_datagen.flow_from_dataframe(
    dataframe=label_df,
    directory = data_directory,
    x_col = 'files',
    y_col = 'labels',
    target_size = (img_width,img_height),
    batch_size = batch_size,
    class_mode = class_mode,
    color_mode = color_mode,
    shuffle = shuffle)

# validation_generator = train_datagen.flow_from_directory(
#     data_directory,
#     target_size = (img_width,img_height),
#     batch_size = batch_size,
#     class_mode = class_mode,
#     color_mode = color_mode,
#     shuffle = shuffle,
#     subset = 'validation')


# model = Sequential()


# model.add(layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(img_height,img_width,1)))
# model.add(layers.Conv2D(64, kernel_size=1, activation='relu'))
# model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2,2),strides = None))
# model.add(layers.Conv2D(128, kernel_size=3, activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2,2),strides = None))
# model.add(layers.Conv2D(256, kernel_size=3, activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2,2),strides = None))

# model.add(layers.Flatten())

# model.add(layers.Dense(500, activation='relu'))
# model.add(Dropout(0.5))

# num_classes = train_generator.num_classes
# model.add(layers.Dense(num_classes, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])

# nb_epochs = 50
# model.fit_generator(
#     train_generator,
#     steps_per_epoch = train_generator.samples // batch_size,
#     validation_data = validation_generator, 
#     validation_steps = validation_generator.samples // batch_size,
#     epochs = nb_epochs)

# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(img_height,img_width,1)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])



# model.fit_generator(
#         train_generator,
#         steps_per_epoch=train_generator.samples // batch_size,
#         epochs=50,
#         validation_data=validation_generator,
#         validation_steps=validation_generator.samples // batch_size)




# model.save_weights('first_try.h5')


# acc = model.evaluate_generator(validation_generator, steps=2,verbose=1)

# print(acc)

# probs = model.predict_generator(validation_generator)
