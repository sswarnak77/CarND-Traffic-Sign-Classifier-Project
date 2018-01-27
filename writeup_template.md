# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/sswarnakar/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43


#### 2. Include an exploratory visualization of the dataset.
The bar chart shows the data distribution of the training data. Each bar represents one class (traffic sign) and how many samples are in the class and we can clearly see that there is huge inbalance between class labels. Due to this imbalance class labels, I was planning to augment more data from external sources but I realized although this dataset is having imbalance class labels but my model performed well so I didn't merge any ore data.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color images dataset is computationally more expensive and has more signal to noise ration compare to grayscale images.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because normalized data can make the training faster and reduce the chance of getting stuck in local optima.
Initially I tried to normalize using given formula : (X_train- 128)/128 but for some weird reason my CNN model was not perfoming well at all.So I decided to tweak this formula a little and came up with (a + ((img - minimum) * (b - a)) / (maximum - minimum)) where a=-.5,b=0.5, minimum=0 and maximum =255.
I also had to shuffle the training dataset for better accuracy.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| output
|:---------------------:|:---------------------------------------------:| 
| Input         		    |   32x32x1	Grayscale Image		  | 
| Convolution 5x5 (6)   |  1x1 stride, valid padding    |  Nx28x28x6  
| RELU					        |												        |
| Max pooling	      	  | 2x2 stride, valid padding     | outputs 14x14x6 
| Convolution 5x5(16)	  | 1x1 stride, valid padding     | ouputs 10x10x16
| RELU                                                                   
| Max pooling           | 2x2 stride, valid padding     | outputs 5x5x16
|Convolution 5x5 (400)  | 1x1 stride, valid padding     | outputs 1x1x400
| Fully connected 1		  | input 400        							| outputs 120 
| RELU				          |       									      |
|Dropout                | Keep Prob = 0.5               |
| Fully connected 2     | input 120                     | outputs 84
| RELU                  |                               |
|Dropout                | Keep Prob = 0.5               |
| Fully connected 3     | input 84                      | outputs 10



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer from Tensorflow instead of SGD.I kept my learning rate to 0.001 and trained the CNN model with 30 Epochs with having a batch size of 100.
I also kept my dropout to 0.5 to prevent overfitting.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95.5%
* test set accuracy of 93.3%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
Initially I tried with simple AlexNet with one convolution layer followed by a pooling and two fully connected layers.

* What were some problems with the initial architecture?
This simple AlexNet was resulting into poor classification of traffic signs as the network architecture was to shallow to classify features like traffic signs.
So I figured out that I needed to go deeper and choose more complex network architecture like VGGNet which can predict complex features in the images.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I directly jumped into VGGNet like architecture and ended up having three convolution layers, two max pooling layers followed by three fully connected layers.
This architecture was perfect to classify more complex features.I also applied dropout in two fully connected layers to prevent the model from overfitting.

* Which parameters were tuned? How were they adjusted and why?
Initially I made the # Epochs to 50 and batch size of 100 but then I realized that my testing and validation accuracies were not changing much over 30 epochs so I decided to end up with 30 # epochs and 100 mini batch size.
I also had to tune my learning rate from 0.01 to 0.001 for gradient descent to perform well rather than getting stuck in local minima.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
I chose to have three convolution layers because I didn't want to make my model too shallow to detect important features.In addition to that having a dropout of 0.5 worked well for me to prevent the model from overfitting.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		    | Stop sign   									| 
| Right-of-way    			| Right-of-way									|
| 30Km/h					      | 30Km/h											    |
| 60 km/h	      	      | Stop sign					 				    |
| Priority road			    | Priority road      						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of the original dataset.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			    | Stop sign   									| 


For the second image ... 

| Probability           |     Prediction                    | 
|:---------------------:|:---------------------------------------------:| 
| 1.0                   | Right-of-way                     | 

For the third image ... 

| Probability           |     Prediction                    | 
|:---------------------:|:---------------------------------------------:| 
| 1.0                   | Yield                     | 
| 0.05                  | Priority road             | 
| 0.1                   | No vehicles               |
| 0.05                  | Speed limit (120km/h)     |  


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


