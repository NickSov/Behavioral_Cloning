# Behavioral Cloning Project Overview
Nick Atanasov | September 9th, 2018

## The goals / steps of this project are the following:

1.	Use the simulator to collect data of good driving behavior
2.	Build, a convolution neural network in Keras that predicts steering angles from images
3.	Train and validate the model with a training and validation set
4.	Test that the model successfully drives around track one without leaving the road

## Model Architecture and Training Strategy

### An appropriate model architecture has been employed
The filter size used was 5x5 followed by 3x3. The depths of the layers ranged between 24 and 64 (model.py lines 135-139). The model includes RELU layers to introduce nonlinearity (model.py lines 135-139), and the data is normalized in the model using a Keras lambda layer (model.py lines 133). Data augmentation was used: images were flipped about the vertical axis  (as were labels), which doubled the amount of data.

My first step was to use a convolution neural network model similar to the Nividia PilotNet architecture as described in the paper Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car (arXiv:1704.07911v1). The architecture is shown in Figure 1. I deemed this network appropriate because it relates to the exact scenario embodied by this project: a self-driving car with camera sensor input. Furthermore, initial tests conducted on LeNet vs PilotNet concluded that the PilotNet architecture was superior.

![PilotNet architecture][PilotNet]
[PilotNet]https://github.com/NickSov/Behavioral_Cloning/blob/master/Pilotnet.png

**Figure 1. PilotNet architecture (arXiv:1704.07911v1)**

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set and an even lower mean squared error on the validation set. This implied that the model was performing well. The validation set verified the performance improvements shown by the training set.


There were definite areas where the car fell off of the track. The hardest part seems to be going around the first dirt portion of side road. To combat this, I include more data samples around that area in order to hopefully train the model to be more familiar with what to do when the embankment is reached. However, additional data actually made the entire model worse, it did not selectively improve the weaknesses of the vehicle. I additionally collected much more date, 5k+ samples of just curves and recover runs. This was in hopes of training the model to perform better around all curves; however, this also did not result in a better outcome.

It seems like less data actually enabled the model to perform better, this is extremely strange in my opinion.

### Attempts to reduce overfitting in the model
The model does not contain dropout layers because drop out layers were tried using the LeNet architecture and they did not improve the outcome. Furthermore, the architecture outlined in the Nvidia paper did not use dropout layers either.

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 62). It can be seen that 20% of the sample set was allocated to a validation set. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track; however, the vehicle only partially completed the run around the track.

### Model parameter tuning
The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 146). It appeared that in most cases the ideal number of epochs was between 2-5 with 3 being the most fruitful.

### Appropriate training data
All of the training was done with the mouse for steering control and "w" on the keyboard for forward throttle.

Training data was chosen to keep the vehicle driving on the road. A wide range of data was used. Many different variants of the data were fed to the network in an effort to train the model effectively. The table below outlines the training runs that were tried:

![training sets][train]
[train]https://github.com/NickSov/Behavioral_Cloning/blob/master/keras_traing.png
