# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 06:36:10 2020

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
from keras import applications




on_windows = False
if on_windows:
    data_directory = 'D:\\Data\\Sketches\\png'
    path_delim = '\\'
else:
    data_directory = '../data/Places_sub'
    path_delim = '/'
    
    
# From https://stackoverflow.com/questions/46717742/split-data-directory-into-training-and-test-directory-with-sub-directory-structu
#image dimensions
img_height =224
img_width = 224


train_datagen = IDG(
    samplewise_std_normalization=True,
    # shear_range = 0.2,
    # zoom_range = 0.2,
    # horizontal_flip = True,
    validation_split = 0.2)

batch_size = 100
class_mode = 'categorical'
color_mode = 'rgb'
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



mobile_net = applications.mobilenet_v2.MobileNetV2(
    include_top=False, 
    weights='imagenet', 
    input_shape=(img_width,img_height,3), 
    pooling='avg')



model = Sequential()
model.add(mobile_net)

for layer in model.layers:
    layer.trainable = False
    
    
# model.add(layers.Flatten())

model.add(layers.Dense(1000, activation='relu'))
model.add(Dropout(0.5))

num_classes = train_generator.num_classes
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])


nb_epochs = 10
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


name = 'mobilenetV2_places_extralayer'
model.save(f'../output/{name}.h5')
plt.savefig(f'../output/{name}.png')


acc = model.evaluate_generator(
    validation_generator, 
    steps=np.floor(validation_generator.n/batch_size),
    verbose=1)

model.summary()

print(acc)


nb_epochs = 10
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


name = 'mobilenetV2_places_extralayer_finetune'
model.save(f'../output/{name}.h5')
plt.savefig(f'../output/{name}.png')


acc = model.evaluate_generator(
    validation_generator, 
    steps=np.floor(validation_generator.n/batch_size),
    verbose=1)

model.summary()

print(acc)