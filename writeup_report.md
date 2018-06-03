# **Behavioral Cloning**


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/nvidia_architecture.png "Nvidia Architecture"
[image2]: ./writeup_images/center.jpg "Center"
[image3]: ./writeup_images/right_recover1.jpg "Recovery Image"
[image4]: ./writeup_images/right_recover2.jpg "Recovery Image"
[image5]: ./writeup_images/right_recover3.jpg "Recovery Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](https://github.com/wwha/CarND-Behavioral-Cloning-P3/blob/master/drive.py) containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* run_video.mp4

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 47-59)

The model includes RELU layers to introduce nonlinearity (code line 50-54), and the data is normalized in the model using a Keras lambda layer (code line 48).

#### 2. Attempts to reduce overfitting in the model

The dropout layers were not included since the final result did not show better than the current one.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 62).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to develop the model of autonomous driving.

My first step was to use a convolution neural network model similar to the [Nvidia Model](https://arxiv.org/pdf/1604.07316.pdf)

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

The final step was to run the simulator to see how well the car was driving around track one.
The speed is changed from 9 kph to 20 kph from the drive.py and the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Final Model Architecture

The final model architecture (model.py lines 50-59) consisted of a convolution neural network with the following layers and layer sizes:
* Image normalization and cropping
* Convolution: 5x5, filter: 24, strides: 2x2, activation: RELU
* Convolution: 5x5, filter: 36, strides: 2x2, activation: RELU
* Convolution: 5x5, filter: 48, strides: 2x2, activation: RELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
* Flatten
* Fully Connected: 100
* Fully Connected: 50
* Fully Connected: 10
* Fully Connected: 1 (output)

Here is a visualization of the architecture

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.


After the collection process, I had 7447 number of data points. I then preprocessed this data by converting the image from BGR to RGB and adding the measurements of the angles for the side camera.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the results. I used an adam optimizer so that manually training the learning rate wasn't necessary.
