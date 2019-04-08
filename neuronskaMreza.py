# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 13:06:59 2019

@author: Sara
"""
#https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten 
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils import np_utils

 

def kreiraj_Model(): #kreiranje modela, region-matrica 28x28
    nm_model = Sequential()
    #ulazni sloj od 784 ulazna neurona
    nm_model.add(Conv2D(28, (3, 3), padding='same', activation='relu', input_shape=(28,28,1)))
    #dodajemo slojeve sa 28 filtera(kernels) 
    #(3x3) window size
    nm_model.add(Conv2D(28, (3, 3), activation='sigmoid')) #matrica 28x28
    nm_model.add(MaxPooling2D(pool_size=(2, 2)))
    nm_model.add(Dropout(0.25)) # dodajemo sloj,dropout ratio 0.25
  
    #dodavanje sloja sa 56 filtera
    nm_model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
    nm_model.add(Conv2D(56, (3, 3), activation='relu'))
    nm_model.add(MaxPooling2D(pool_size=(2, 2)))
    nm_model.add(Dropout(0.25))
  
    nm_model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
    nm_model.add(Conv2D(56, (3, 3), activation='relu'))
    nm_model.add(MaxPooling2D(pool_size=(2, 2)))
    nm_model.add(Dropout(0.25))
  
    nm_model.add(Flatten())
    #128 neurona u skrivenom sloju
    nm_model.add(Dense(128, activation='relu'))
    nm_model.add(Dropout(0.5))
    #10 neurona na izlaznom sloju
    nm_model.add(Dense(10, activation='softmax'))
  
    return nm_model

def obucavanjeMreze():
    (X_train, y_train), (X_test, y_test) = mnist.load_data() 
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    X_train /= 255
    X_test /= 255
    
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)


    nm_model = kreiraj_Model() #pozivamo funkciju za kreiranje modelaa
    #obucavanje neuronske mreze
    epochs = 10
    verbose = 1
    batch_size=256
    nm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    nm_model.fit(X_train, Y_train, batch_size, epochs, verbose, shuffle=False) 
    score = nm_model.evaluate(X_test, Y_test, verbose=0)          
    print(score)
  
    #cuvanje modela u hdf5 fajluu
    
    nm_model.save_weights("model.h5")

    return 1,nm_model







