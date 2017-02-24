# **Traffic Sign Recognition**


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Complete codes and results for this project can be found in the  [`Traffic_Sign_Classifier.ipynb`](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### 1. Data Set Summary & Exploration

#### 1.0 - Where in ipynb?  
Please refer to *Step 1: Dataset Summary & Exploration* in [`Traffic_Sign_Classifier.ipynb`](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).  
Pandas library was used to calculate summary statistics of the traffic signs data set and matplotlib library to visualize basic statistics of the data set.

#### 1.1 - Data set statistics  
Please refer to *1.1: Basic Summary of the Data Set Using Python, Numpy and/or Pandas* in [`Traffic_Sign_Classifier.ipynb`](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).
* Size of training set (`X_train`): 34799
* Size of validation set (`X_validation`): 4410
* Size of testing set (`X_test`): 12630
* Shape of a traffic sign image (width - height - channels): 32 by 32 by 3
* Number of unique classes/labels in the data set: 43 (integer label from 0 to 42)


#### 1.2 - Include an exploratory visualization of the dataset and identify where the code is in your code file.
Please refer to *1.2: Exploratory visualization of dataset* in [`Traffic_Sign_Classifier.ipynb`](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

* Histogram of label counts: 2nd cell of **1.2**
    ![labels count distribution][./report_fig/11.png]

 * Sample of four images per class: 5th cell of **1.2**
     ![sample traffic sign images][./report_fig/12.png]
Sample images of three arbitrarily selected labels are presented here. Please refer to the .ipynb for full sample images.

* Dark/bright images: 6th cell of **1.2**
     ![bright/dark sign images][./report_fig/12.png]
As can be seen from the above images, traffic sign images are composed of some extremely bright and dark images. These conditions may affect the traffic sign classification process. For this reason, image preprocessing was conducted, but turned out that even with such extreme brightnesses, feeding the original images result in better classifications. This will be discussed in later section (**X.X**).


### 2. Design and Test a Model Architecture

#### 2.0 - Where in ipynb?
Please refer to *Step 2: Design and Test a Model Architecture* in [`Traffic_Sign_Classifier.ipynb`](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).  

#### 2.1 Pre-process the Data Set
Please refer to *2.1: Pre-process the Data Set* in
[`Traffic_Sign_Classifier.ipynb`](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

As mentioned earlier, original images varies significantly in their brightness. Thus, a pre-processing was conducted and fed for training with Neural Network. The pre-processing method was normalization. However, the training/testing results turned out to be worse with the normalized images. For this reason, normalization was discarded.  
Possible reason for worse traning with normalized images could be due to the colors of traffic signs (red, blue, and yellow) contributing significantly to their classifications.  
Below are images of arbitrary traffic signs before and after normalization.

 ![normalization][./report_fig/21.png]
*TODO: fix the unmatching images before/after processing*

#### 2.2 Iput Data
Please refer to *2.2: Input Data* in [`Traffic_Sign_Classifier.ipynb`](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).
Training, validation, and testing data were provided pre-split in separate files. Conveniently, there was no need to conduct data splitting process. If I had to, I would have used an `ssklearn.model_selection.train_test_split`.  
Data were shuffled with `sklearn.utils. shuffle` to avoid overfitting.  
In the future work, I will prepare and feed more training/validating/testing images collected from outside source or processed-copy of the original images.


#### 2.3 Model Architecture (Modified LeNet)
Please refer to *2.3: Model Architecture (Modified LeNet)* in [`Traffic_Sign_Classifier.ipynb`](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

My final model is a slightly modified version of a LeNet. Features added are Local Response Normalization and Dropouts. Final network architecture looks like the following:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Local Resps Normaliz 	|                               				|
|                                                          				|
| Convolution 3x3	    | etc.      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Flattening        	| outputs 400                      				|
|                                                          				|
| Fully connected		| output 120        							|
| RELU					|												|
| Dropout   			|												|
|                                                          				|
| Fully connected		| output 84        				    			|
| RELU					|												|
| Dropout   			|												|
|                                                          				|
| Fully connected		| output 10         							|
|                                                          				|
| Output				| logits with size 10(number of labels)         |



#### 2.4 Train, Validate, and Test Model
Please refer to *2.4: Train, Validate, and Test Model* in
[`Traffic_Sign_Classifier.ipynb`](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

Parameters implemented:
- Optimizer: Adaptive Moment Estimation(Adam) Optimizer
    [Good optimizer reference](http://sebastianruder.com/optimizing-gradient-descent/index.html#adam)
- Batch Size: 100
- Epoch Size: 15
- Dropout Rate: 0.5
- Weight initialization: Truncated Normal Distribution with mean: 0 and standard deviation: 0.1
- Learning Rate: 0.1

Below are the steps taken for training, validating, and testing:
1. A series of parameter tunings were conducted with trial-and-error with batch size `BATCH_SIZE`, epoch size `EPOCHS`, dropout rate  `dropout`, and learning rate `rate`. Since trial-and-error was somewhat unsystematic method for finding optimal parameters, in the future work I will try to implement a more systematic approach for parameter tuning similar to Gradient ???? .  
2. Training dataset was fed to the model for training.
3. Validation datasets were fed `EPOCH` number of times and prediction accuracies were measured.
4. Testing Dataset was fed and prediction accruacy was measured.

#### 2.5 Accuracies of the Model
Please refer to the *result boxex after running codes in* *2.4: Train, Validate, and Test Model* in
[`Traffic_Sign_Classifier.ipynb`](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

Reported accuarcies are as followed:
* training set:
* validation set:
* test set:

An enhanced version of LeNet was used for traffic sign classification. This did a fairly good job in predicting traffic sign with an accuracy of xxx%. I deemed this to be a good model for xxx because of LeNet's proven history of good performance. Additional features such as dropouts and batch mean xxx were intended add additional overfitting prevention and xxxx. However as mentioned earlier, implementing a more systematic parameter optimization method will guarantee a much higher model prediction accuracy.


### 3. Test a Model on New Images
#### 3.0 Where in ipynb?
Please refer to *Step 3: Test a Model on New Images* in [`Traffic_Sign_Classifier.ipynb`](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).  

#### 3.1 Image Specifications
Five random German Traffic Signs were capture from Google image search. Five images are displayed below:
    ![Fice German Traffic Images][./report_fig/31.png]
Label for each image was hard coded by comparing with sample test images from above. From the top, numerical label for each image is 38, 13, 18, 12, and 28  

Images chosen are easy to classify for them having vivid colors and shapes. I assumed that the last image from the list will be the most difficult to classify due to the complex nature of the sign; an image of an adult holding hand of a child.

#### 3.2 Prediction Results

Model prediction of each test image was reported in a list of top 5 likely labels along with % likelyhood of each label.
Here are the results of the prediction:

| Ground Truth		     |     Prediction	        		| Percentage            |
|:---------------------:|:---------------------------------:|:---------------------:|
| 38      	        	| 38   				    			| 95.731%               |
| 13     	    		| 13 								| 100.0%                |
| 18					| 18								|  99.81%               |
| 12	      	    	| 12				 				|  99.996%              |
| 28		        	| 28      							|  99.653%              |


Surprisingly, the model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 89.9%
