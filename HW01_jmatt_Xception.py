# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 07:40:19 2020

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
from keras import Model
from keras.models import load_model




on_windows = False
if on_windows:
    data_directory = 'D:\\Data\\Sketches\\png'
    path_delim = '\\'
else:
    data_directory = '../data/places_200'
    path_delim = '/'
    
    
# From https://stackoverflow.com/questions/46717742/split-data-directory-into-training-and-test-directory-with-sub-directory-structu
#image dimensions
img_height = 299
img_width = 299


train_datagen = IDG(
    samplewise_std_normalization=True,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
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


num_classes = train_generator.num_classes

load_from_file = True
if load_from_file:
    model = load_model('../output/Xception_places_200_FE.h5')
    for layer in model.layers:
        layer.trainable=False
else:

    xception = applications.xception.Xception(
        include_top=False, 
        weights='imagenet', 
        input_shape=(299,299,3), 
        pooling='avg')
    
    for layer in xception.layers:
        layer.trainable = False
    
    
    
    fc = layers.Dense(num_classes,activation='softmax')(xception.output)
    
    model = Model(inputs = xception.input,outputs=fc)
    

    model.summary()

# model = Sequential()
# model.add(xception)

    

# model.add(layers.Dense(1000, activation='relu'))
# model.add(layers.Dense(1000, activation='relu'))
# model.add(Dropout(0.5))

# model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics = ['accuracy'])
    
    
    loops = 1
    for i in range(loops):
        print ('\n\nEPOCH SET {}'.format(i))
        nb_epochs = 2
        history = model.fit_generator(
            train_generator,
            steps_per_epoch = train_generator.samples // batch_size,
            validation_data = validation_generator, 
            validation_steps = validation_generator.samples // batch_size,
            epochs = nb_epochs)
        
    name = 'Xception_places_200_FE'
    model.save(f'../output/{name}.h5')



#number of layers counting from the end that will be trainable
num_trainable = [11, 32, int(len(model.layers)/2), int(len(model.layers)*3/4)]
for ind,nt in enumerate(num_trainable):
    for layer in model.layers[-1*nt:]:
        layer.trainable = True
        
        
    model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics = ['accuracy'])
    
    model.summary()
    
    loops = [1,1,2,3]
    for i in range(loops)[ind]:
        print ('\n\n{} Trainable Layers: EPOCH SET {}'.format(nt,i))
        nb_epochs = 2
        history = model.fit_generator(
            train_generator,
            steps_per_epoch = train_generator.samples // batch_size,
            validation_data = validation_generator, 
            validation_steps = validation_generator.samples // batch_size,
            epochs = nb_epochs)
        
    name = 'Xception_places_200_FT_NT{}'.format(nt)
    model.save(f'../output/{name}.h5')





# # Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')



# plt.savefig(f'../output/{name}.png')



# acc = model.evaluate_generator(
#     validation_generator, 
#     steps=20,
#     verbose=1)

# model.summary()

# print(acc)


# for layer in model.layers:
#     layer.trainable = True
    
    

# loops = 5
# for i in range(loops):
#     print ('\n\nEPOCH SET {}'.format(i))
#     nb_epochs = 2
#     history = model.fit_generator(
#         train_generator,
#         steps_per_epoch = train_generator.samples // batch_size,
#         validation_data = validation_generator, 
#         validation_steps = validation_generator.samples // batch_size,
#         epochs = nb_epochs)
#     name = 'Xception_places_200_FE'
#     model.save(f'../output/{name}.h5')



# # # Plot training & validation accuracy values
# # plt.figure()
# # plt.plot(history.history['acc'])
# # plt.plot(history.history['val_acc'])
# # plt.title('Model accuracy')
# # plt.ylabel('Accuracy')
# # plt.xlabel('Epoch')
# # plt.legend(['Train', 'Test'], loc='upper left')


# name = 'Xception_bigout_places_995_finetune'
# model.save(f'../output/{name}.h5')
# # plt.savefig(f'../output/{name}.png')



# acc = model.evaluate_generator(
#     validation_generator, 
#     steps=20,
#     verbose=1)

# model.summary()

# print(acc)