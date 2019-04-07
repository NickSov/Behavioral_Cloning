# Behavior Cloning Project | Term 1 | Udacity
# Nicholas Atanasov
# September 13th, 2018


import csv
import zipfile
import numpy as np
import os
import sys
import imp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
import cv2
import tensorflow as tf

##### Data Ingest and Array Creation #####

### Read in image location and label data ###

lines = []

with open('driving_log.csv') as drivingLog:
	readCSV = csv.reader(drivingLog)
	for line in readCSV:
		lines.append(line)

print('Reading images... \n')

### Create image and label arrays ###

corFact = 0.2
imageList = []
steerAngleArray =[]

for line in lines:

	### Path extraction

    sourcePathCenter = line[0]
    fNameCenter = sourcePathCenter.split('/')[-1]
    sourcePathLeft = line[1]
    fNameLeft = sourcePathLeft.split('/')[-1]
    sourcePathRight = line[2]
    fNameRight = sourcePathRight.split('/')[-1]

    ### Path concatenation

    currentPathCenter = 'IMG/' + fNameCenter
    currentPathRight = 'IMG/' + fNameRight
    currentPathLeft = 'IMG/' + fNameLeft

	### Read in images from all cameras

    centerImage = cv2.imread(currentPathCenter)
    rightImage = cv2.imread(currentPathRight)
    leftImage = cv2.imread(currentPathLeft)
    centerLabel = float(line[3])
    rightLabel = float(line[3])-corFact
    leftLabel = float(line[3])+corFact

	# center camera data

    imageList.append(centerImage)
    steerAngleArray.append(centerLabel)

	# left camera data

    imageList.append(leftImage)
    steerAngleArray.append(leftLabel)

	# right camera data

    imageList.append(rightImage)
    steerAngleArray.append(rightLabel)

	### Image augmentation : flip images - all cameras

	# center camera data flipped

    imageList.append(cv2.flip(centerImage,1))
    steerAngleArray.append(centerLabel*-1.0)

	# left camera data flipped

    imageList.append(cv2.flip(leftImage,1))
    steerAngleArray.append(leftLabel*-1.0)

	# right camera data flipped

    imageList.append(cv2.flip(rightImage,1))
    steerAngleArray.append(rightLabel*-1.0)

trainInput = np.array(imageList)
labelInput = np.array(steerAngleArray)

print(trainInput.shape)
print(labelInput.shape)

print('Done creating image and label arrays. \n')

##### Nueral Net Architecture #####

# Architecture is modeled after the work of Nivida
# Publication: "Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car"
# Citation: arXiv:1704.07911v1

print('Initiating training via Keras \n')

from keras.models import Sequential
from keras.layers import Cropping2D, Flatten, Activation, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape =(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
model.add(Dropout(0.9))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(0.9))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

##### Network Training and Output #####

### Network training specification

epochCount = 3
model.compile(loss='mse',optimizer='adam')
history_object = model.fit(trainInput, labelInput, validation_split=0.2, shuffle=True, nb_epoch=epochCount, verbose=1)

### Save the model

model.save('model.h5')

print('model saved')

exit()
