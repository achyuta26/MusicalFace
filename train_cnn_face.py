#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 14:03:22 2018

@author: achyutajha
"""

import cv2
import os
from os import walk

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation,Flatten
from keras.optimizers import Adam
import numpy as np


basedir='/Users/achyutajha/Documents/PSU Study Mat/Fall-II/Deep Learning/Project/Data/KaggleData/fer2013/'
training_path = basedir + 'Training'
testing_path = basedir + 'PublicTest'
num_classes = 7

################## Data Collection and preprocessing stages ##################

def img_to_matrix(imagePath):
    image=cv2.imread(imagePath)
    image=cv2.resize(image, (48,48))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def prepare_data(path):
    X=[]
    Y=[]
    labels = []
    for (_, dirnames, _) in walk(path):
        labels.extend(dirnames)
    for label in labels:
        for root, dirs, files in os.walk(os.path.abspath(path+'/'+label)):
            for file in files:
                imagePath=root +'/'+ file
                image=img_to_matrix(imagePath)
                X.append(image)
                Y.append([int(label)])
    return X,Y


def preprocess(X,Y):
    flat_X = np.array(X)
    flat_Y = np.array(Y)
    flat_X = flat_X.astype('float32')
    flat_X/=255
    flat_Y = keras.utils.to_categorical(flat_Y, num_classes)
    return flat_X,flat_Y


X_train,Y_train = prepare_data(training_path)
X_test,Y_test = prepare_data(testing_path)


flat_X_train,flat_Y_train = preprocess(X_train,Y_train)
flat_X_test,flat_Y_test = preprocess(X_test,Y_test)


flat_X_train = flat_X_train.reshape(flat_X_train.shape[0], 48, 48, 1)
flat_X_test = flat_X_test.reshape(flat_X_test.shape[0], 48, 48, 1)



from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization



################### Build the neural network ##################

model_8 = Sequential()

#1st block
model_8.add(Conv2D(64, (3,3), strides = (1,1), padding='same',
                 input_shape = flat_X_train.shape[1:],
                   kernel_initializer="lecun_uniform",
                   kernel_regularizer=regularizers.l2(0)))
model_8.add(BatchNormalization())
model_8.add(Activation('tanh'))
model_8.add(MaxPooling2D(pool_size=(2, 2)))
model_8.add(Dropout(0.25))

#2nd block
model_8.add(Conv2D(128, (5,5), strides = (1,1),kernel_regularizer=regularizers.l2(0)))
model_8.add(BatchNormalization())
model_8.add(Activation('tanh'))
model_8.add(MaxPooling2D(pool_size=(2, 2)))
model_8.add(Dropout(0.25))


#3rd block
model_8.add(Conv2D(512, (3,3), strides = (1,1), padding='same',kernel_regularizer=regularizers.l2(0)))
model_8.add(BatchNormalization())
model_8.add(Activation('tanh'))
model_8.add(MaxPooling2D(pool_size=(2, 2)))
model_8.add(Dropout(0.5))

#4th block
model_8.add(Conv2D(512, (3,3), strides = (1,1), padding='same',kernel_regularizer=regularizers.l2(0)))
model_8.add(BatchNormalization())
model_8.add(Activation('tanh'))
model_8.add(MaxPooling2D(pool_size=(2, 2)))
model_8.add(Dropout(0.1))


#5th block
model_8.add(Flatten())
model_8.add(Dense(256,kernel_initializer="lecun_uniform"))
model_8.add(BatchNormalization())
model_8.add(Activation('relu'))
model_8.add(Dropout(0.5))
model_8.add(Dense(512,kernel_initializer="lecun_uniform"))
model_8.add(BatchNormalization())
model_8.add(Activation('relu'))
model_8.add(Dropout(0.5))
model_8.add(Dense(num_classes))
model_8.add(Activation('softmax'))

model_8.summary()


learning_rate = .001
# sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

model_8.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adam(lr=learning_rate),
              metrics=['accuracy'])


from keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=7, \
                          verbose=1, mode='auto')
callbacks = [earlystop]


batch_size = 128
epochs = 35
history_8 = model_8.fit(
    flat_X_train, flat_Y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks,
    validation_data=(flat_X_test,flat_Y_test))


model_8.save('./face_cnn_model.h5')