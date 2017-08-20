# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_imgs/viz_1.png "Sample Training Set"
[image2]: ./writeup_imgs/viz_2.png "Sample Training Set"
[image3]: ./writeup_imgs/viz_3.png "Sample Training Set"
[image4]: ./writeup_imgs/viz_4.png "Sample Training Set"
[image5]: ./writeup_imgs/1.png "Traffic Sign 1"
[image6]: ./writeup_imgs/2.png "Traffic Sign 2"
[image7]: ./writeup_imgs/3.png  "Traffic Sign 3"
[image8]: ./writeup_imgs/4.png  "Traffic Sign 4"
[image9]: ./writeup_imgs/5.png  "Traffic Sign 5"

## Rubric Points
Analysis and consideration of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) and how I addressed them.

---

### Data Set Summary & Exploration

##### 1. Basic summary of the data set:

* The size of training set is: 34,799 examples
* The size of the validation set is: 4,410 examples
* The size of test set is: 12,630 examples
* The shape of a traffic sign image is: 32 x 32 x 3
* The number of unique classes/labels in the data set is: 43 Classes

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Random images from the dataset. 

![alt text][image1] ![alt text][image2]
![alt text][image3] ![alt text][image4]

### Design and Testing of Model Architecture

#####  Preprocessing

1. First step was to shuffle the data to make it random. Used the <code>shuffle()</code> method from <code>sklearn.utils</code>
2. Normalized the data for better training performance to a value of 0 to 1. This allows for faster computations and better results.

##### Model Architecture

My final model consisted of the following layers:

| Layer                | Description                                   | 
| :------------------: | :-------------------------------------------: | 
| Input                | 32x32x3 RGB image                             | 
| Convolution 5x5      | 1x1 stride, valid padding, outputs 28x28x64   |
| RELU                 |                                               |
| Max pooling          | 2x2 stride,  outputs 14x14x64                 |
| Convolution 5x5      | 1x1 stride, valid padding, outputs 10x10x128  |
| RELU                 |                                               |
| Max pooling          | 2x2 stride,  outputs 5x5x128                  |
| Flatten              | Outputs 3200                                  |
| Fully connected      | Outputs 240                                   |
| RELU                 |                                               | 
| Fully connected      | Outputs 120                                   | 
| RELU                 |                                               | 
| Fully connected      | Outputs 43, which is number of Classes/Labels | 
| Softmax              | Create probabilities from Logits              |
| Cross_Entropy        | Produces the One Hot Encoded Predictions      |

To train the model I used a learning rate of <code>0.0009</code>, with 40 epochs,  and a batch size of 256. Used cross_entropy to predict while training, then used tensorflow's <code>tf.reduce_mean()</code> to calculate the loss. Then using <code>AdamOptimizer</code>, backpropagation was performed and the model was updated.

My solution was based heavily on the LeNET model architecture. From my previous Deep Learning Foundations, I realized this was a good model. To achieve the 0.93+ validation accuracy, I basically tried an iterative process. By using more epochs, larger batch size, small learning_rate, and adding layers (going deeper), I was able to get a good validation.

My final model results were:
* validation set accuracy: 93%
* test set accuracy : 92.4%

An iterative approach was chosen:
* The first architecture chosen was almost identical to the LeNET architecture
* The LeNET architecture didn't achieve more than 85% validation accuracy.
* To achieve 93% validation accuracy, I increased the output depth of each convolutional layer, plus added an additional fully connected layer
* The epochs and batch_size were modified iterative until best accuracy achieved
* By adding dropout, the network might improve more

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

Here are the results of the prediction:

| Image            | Prediction       | 
| :---------------:| :---------------:| 
| Slippery Road    | Slippery Road    | 
| 60 km/h          | 60 km/h          |
| Keep right       | Keep right       |
| Children Crossing| Children Crossing|
| Road work        | Road work        |


The model was able to correctly guess 5 of the 5 traffic signs (with an accuracy of 100% for each), which gives an accuracy of 100%. This compares favorably to the accuracy on the test set, which was 93%. The test set has 12,630 examples, which increases the chances of failing more than a set of 5 images.

The code for making predictions on my final model is located in the 9th code cell of the Python notebook.

For the first image, the model is sure that this is a Slippery Road sign (probability of 0.99), and the image does contain a slippery road sign. The top five soft max probabilities were

| Probability         	| Prediction       | 
| :------------------: | :---------------:| 
| 0.99                 | Slippery Road    | 
| 0.99                 | 60 km/h          |
| 1.0                  | Keep right       |
| 0.98                 | Children Crossing|
| 1.0                  | Road work        |



