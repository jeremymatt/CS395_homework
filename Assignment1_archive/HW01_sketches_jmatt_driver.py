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
from keras.models import load_model




on_windows = False
if on_windows:
    data_directory = 'D:\\Data\\Sketches\\png'
    path_delim = '\\'
else:
    data_directory = '../data/Sketches/png'
    path_delim = '/'



# From https://stackoverflow.com/questions/46717742/split-data-directory-into-training-and-test-directory-with-sub-directory-structu
#image dimensions
img_height = 224
img_width = 224


train_datagen = IDG(
    featurewise_std_normalization=True,
    shear_range = 0.25,
    zoom_range = 0.25,
    rotation_range = 45,
    horizontal_flip = True,
    vertical_flip = True,
    validation_split = 0.2)

batch_size = 100
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


load_from_file = True
if load_from_file:
    model = load_model('../output/jmatt_arch_sketches_orig.h5')
else:
    
    model = Sequential()
    
    
    model.add(layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(img_height,img_width,1)))
    model.add(layers.Conv2D(32, kernel_size=1, activation='relu'))
    model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2),strides = None))
    model.add(layers.Conv2D(128, kernel_size=3, activation='relu'))
    # model.add(layers.Conv2D(64, kernel_size=1, activation='relu')) #
    # model.add(layers.Conv2D(128, kernel_size=3, activation='relu')) #
    model.add(layers.MaxPooling2D(pool_size=(2,2),strides = None))
    model.add(layers.Conv2D(256, kernel_size=3, activation='relu'))
    # model.add(layers.Conv2D(128, kernel_size=1, activation='relu')) #
    # model.add(layers.Conv2D(256, kernel_size=3, activation='relu')) #
    model.add(layers.MaxPooling2D(pool_size=(2,2),strides = None))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    
    num_classes = train_generator.num_classes
    model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

nb_epochs = 2
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = nb_epochs)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')


name = 'jmatt_arch_sketches_orig_tune'
model.save(f'../output/{name}.h5')
plt.savefig(f'../output/{name}.png')



acc = model.evaluate_generator(
    validation_generator, 
    steps=np.floor(validation_generator.n/batch_size),
    verbose=1)

print(acc)

probs = model.predict_generator(validation_generator)
