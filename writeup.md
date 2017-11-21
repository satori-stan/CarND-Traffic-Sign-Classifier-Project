#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./static/visualization.png "Visualization"
[image2]: ./static/example.png "Random Example: Right of way at the next intersection (class 11)"
[image3]: ./static/example_augmented1.png "Random Example after augmentation"
[image4]: ./static/example_augmented2.png "Random Example after augmentation bis"
[image5]: ./static/visualization2.png "Train data histogram after augmentation"
[image6]: ./static/example_preprocessed.png "Random example after preprocessing"
[image7]: ./static/training_loss.png "Training loss evolution"
[image8]: ./images/1.png "Traffic Sign 1"
[image9]: ./images/13.png "Traffic Sign 2"
[image10]: ./images/17.png "Traffic Sign 3"
[image11]: ./images/21.png "Traffic Sign 4"
[image12]: ./images/23.png "Traffic Sign 5"
[image13]: ./images/25.png "Traffic Sign 6"
[image14]: ./images/28.png "Traffic Sign 7"
[image15]: ./images/30.png "Traffic Sign 8"
[image16]: ./images/40.png "Traffic Sign 9"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/satori-stan/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set. Basically take advantage of the "shape" property of a numpy array:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

Here is a random example from the data that we will be following through the calculations ...

![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to augment the dataset. As can be seen by the exploratory visualization (histogram), some classes are severely underrepresented (~180-240) while others have a significant number of examples (~1800-2010). Reading up on possible augmentation techniques I found shearing, rotation, translation, color modifications and occlusion to be commonplace.

Translation moves the image around in the same space to account for the fact that not all images sent to the network will be centered. Shearing and rotation account for different points of view. Zooming accounts for changes in the size of the sign. In a self-driving car product-line not all image capture devices may be in the same position (height and angle relative to the road) and not all captures may happen from the same GPS coordinates (i.e. position relative to the sign), so the sign might look slightly different.

Color modification will help with identifying the traffic sign in different times of the day or weather conditions.

Occlusion will account for other objects hiding part of the traffic sign. These objects may be other signs, pedestrians, trees, stickers, graffiti, etc.

For my submission I decided to use all except occlusion to simplify the solution, although color modification was part of the preprocessing step and not the data augmentation. After several searches on how to augment a dataset, I found [code](https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc) that applies these transformations using the OpenCV library and integrated it to my project. The functions in question apply random transformations to an image within the provided limits. I set the limits at +/- 45° of rotation, 5 pixels of shear and 10 pixels of translation.

Here are two different transformations of the random example image:
![alt text][image3]
![alt text][image4]

The augmentation was only applied to the Train data set since the Validation and Test sets are meant only to test the robustness of the network that should already take into account all the differences in the image.

Here is the example count per class after augmentation:

![alt text][image5]

After augmentation, the least amount of examples for a class is now 1170.

My code saves and loads the augmented dataset to avoid the lengthy augmentation process. One possible improvement would be to augment not only the underrepresented classes, but all classes to improve the accuracy of the network for new images.

The preprocessing step is applied to all images, regardless of whether they are from the train, validation, test or new datasets. This ensures all images have similar numerical values for the calculations.

Preprocessing has three steps: color modification in the form of intensity scaling, grayscaling as a measure to reduce the input size for performance and normalization to aid in the performance of the optimization stage of the training.

The color modification, or rather intensity rescaling stretches the pixel values between the 2nd and 98th percentile of the intensities, so that the image appears brighter and with better contrast.

I convert the images to grayscale because having less channels requires less parameters for the initial convolutions, saving some memory and making the calculations faster. Intuitively though, a human would be able to identify the traffic signs in grayscale as well as would the color versions so I didn't think much of information was being discarded.

Here is an example of a traffic sign image after preprocessing.

![alt text][image6]

As a last step, I normalized the image data because optimization algorithms tend to do best with normalized data since all values are within a small range.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer           |     Description                             |
|:---------------:|:-------------------------------------------:|
| Input           | 32x32x3 Grayscale image   			|
| Convolution 5x5 | 1x1 stride, same padding, outputs 32x32x6   |
| RELU            |						|
| Convolution 5x5 | 1x1 stride, same padding, outputs 32x32x6   |
| RELU            |						|
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x16 |
| RELU            |						|
| Max pooling     | 2x2 stride,  outputs 14x14x16               |
| Dropout         | 50% keep probability (training only)        |
| Fully connected | 120 units + bias and RELU                   |
| Dropout         | 50% keep probability (training only)        |
| Fully connected | 84 units + bias and RELU                    |
| Dropout         | 50% keep probability (training only)        |
| Fully connected | 43 units (classes) + bias                   |
| Softmax         | To get the one hot prediction probabilities |
 
A number of architectures and layer sizes were tested for prediction performance and training time. The exercise started with a vanilla LeNet and from there I added convolutions, removed fully connected layers and increased the layer sizes. Additional tests where information skipped layers (a la Inception or ResNet) were attempted as well. The intuition is always that more layers are better, prefer convolutions over fully connected layers and avoid reducing your dataset too much on the initial layers.

The final architecture has three convolution layers without maxpooling between them and the first two have VALID padding, precisely to avoid data reduction and catch as many features as possible. The RELU layers are mandatory since without them, the convolutions are just linear transformations. The idea to keep the number of channels in the first two convolutions comes from ResNet and seems to refine the feature mapping of the first layer.

After the last convolution, I introduce Max Pooling. By now the features in the images have been extracted and we can squeeze the information in them to avoid a huge initial fully connected layer, which would negatively impact the execution performance of the network.

Dropout is introduced at this stage as well. Previous attempts using dropout in the whole model didn't prove effective: it would increase the noise of the cost calculation, slowing down the learning process and not showing significant improvements in the accuracy over the validation dataset.

The three Fully connected layers decrease the number of units in the same steps as the original LeNet did.

Finally, the Softmax layer transforms the probabilities to the predicted class.

As a sidenote, the placeholder initialization was changed from the Truncated Normal used in the LeNet-Lab to the Xavier/2 recommended by the C231n Stanford lectures. Curiously, the training loss plateaued sometimes before decreasing again and the validation accuracy decreased before it increased, but for this particular model, proved a good fit.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer (same as in the LeNet-Lab, but in all the literature, it proved to be the best algorithm). The learning rate came down to 1e-5. A higher learning rate apparently failed to converge since the loss didn't present a steady decline. I did play around with decaying learning rates, but dropped them when they didn't seem to provide much benefit. Although it very well might have been because I wasn't using it properly.

My model was trained for 1400 epochs with batch size of 256 examples, which took 255 minutes to train. I do think that more epochs are needed to properly train the model, even if the accuracy wouldn't improve much. For the total 74217 examples in my augmented training set, at least 300 epochs are necessary to go once over every example, and that is without taking into account the randomness introduced by the batch gradient descent, which means we don't really know if our training loop has actually gone over all the examples in the training set.

Here is the evolution of the training loss and validation accuracy:

![alt text][image7]

Running the model for a long time was a big problem because the Jupyter session kept being automatically closed after some time of inactivity. Because of this, my training code will first loads weights from previous sessions if found. In the end this feature was not very useful because while it helped with seeing results of more epochs, I data for the loss and accuracy graphs.

After some more reading I found out that it is possible to instruct TensorFlow to store checkpoints after so many hours of runtime or for specific epochs of the training process. It is also possible to record summaries to display in TensorBoard for the same graphs that I was plotting in Jupyter. Definitely will use those in future projects.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 85.2%
* validation set accuracy of 95.8%
* test set accuracy of 95.3%

The first architecture, as suggested was the LeNet architecture used for the LeNet lab project.
Adding dropout to a LeNet architecture, using a leaning rate 0.0001 and increasing the Epochs to 500 pushed Validation accuracy to almost 95%. Without dropout, the system overfits the training set and plateaus at around 90%. This suggests that the model overfits and requires regularization, which dropout provides. Although it was still not enough to get good results in the test set or the new images.
To improve the results after that, I tried adding convolutions of different filter sizes with a pooling operation in between. Next came increasing the feature count of the convolutions. After seeing only minimal improvements, I then attempted with a single inception layer. Finally I started removing the pooling operations and reducing again feature counts of the convolutions to reduce running times.
It seems that even though the gains from different models were minimal, what really improved the validation and new image accuracy was allowing enough time for the network to learn. With the final model, it was slow to get past the 93% validation accuracy but the loss plot still appeared to have space before plateauing.
All hyperparameters were adjusted. Learning rate quickly showed that 1e-5 was a likely optimal value. Higher rates tended to lead to underfitting, showing that the optimization process was not reaching the minimums. Lower rates significantly increased the training required.
For most of my tests, 200 epochs took the validation accuracy over 80% and by 500 it was around 93%, but for the test set validation, it always fell a couple of percentage points. I ended up with 1400 epochs to allow the model to strengthen the learnt features.
I also increased the batch size from 64 to 256. With the augmented training set I needed either more epochs or more examples per epoch to allow the model to learn them all. As I increased the batch size, I kept a close eye on the GPU usage to make sure I was using it efficiently.
Finally, the dropout rate was set to 50% during training as recommended by the paper which introduced it. Since it is only applied to the fully connected layers, I had only one rate.

There are a couple of things that I would change for a work project. Given a dedicated computer, I would allow the training to run for full days or weeks. With that timeframe in mind, I definitely would make the network larger. I would also first train the network on the provided training set without dropout or augmented examples before adding them, which I think would mean that all improvement after the initial training would be towards better generalization of the model.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image15] ![alt text][image16] ![alt text][image12] 
![alt text][image11] ![alt text][image10] ![alt text][image8]
![alt text][image9] ![alt text][image14] ![alt text][image13]

The first image might be difficult to classify because in general, the network has difficulty with the internal features of the signs. This issue is most clear with the third and ninth images.

The second, third and fifth images add difficulty because they are not properly centered, so there is some translation at play that the augmented dataset should cover. The fourth image also presents some shearing.

In general, the images that I found online were not of the traffic sign alone so I had to trim them. After trimming, some of them were still of a different size that the necessary 32x32 so I had to resize them. After the two operations, the aspect ratio of the images to the canvas size still didn't match the original set, which made it difficult for the network. Some of the images had to be stretched to fill the 28x28 space of the original images before they were correctly classified.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                 | Prediction                  | 
|:---------------------:|:---------------------------:| 
| Beware of ice/snow    | Beware of ice/snow          | 
| Roundabout mandatory  | Roundabout mandatory        |
| Slipper road          | Dangerous curve to the left |
| Double curve          | Beware of ice/snow          |
| No entry              | Stop                        |
| Speed limit (30 km/h) | Speed limit (30 km/h)       |
| Yield                 | Yield                       |
| Children crossing     | Children crossing           |
| Road work             | Road work                   |

The model was able to correctly guess 6 of the 9 traffic signs, which gives an accuracy of 66%. This doesn't compare favorably to the accuracy on the test set of the original data set. I expect the training set accuracy should increase (so that the network can get better used to the augmented images and normalization) before some of these images can be correctly classified.

Some of the incorrectly classified signs in the submitted notebook were correctly classified in previous attempts, for example traffic sign 5 which is pretty recognizable for humans regardless of the translation present in the image.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 26th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability |     Prediction               | 
|:-----------:|:----------------------------:| 
| .60         | Beware of ice/snow           |
| .33         | Slippery Road                |
| .03         | Bicycles crossing            |
| .01         | Dangerous curve to the right |
| .00         | Wild animals crossing        |


For the second image, the model predicted the roundabout sign but with a lower certainty (only 0.4). The fact that it is still 15 percentage points over the closest match tells me the prediction is still accurate.

| Probability |     Prediction               | 
|:-----------:|:----------------------------:| 
| .40         | Roundabout mandatory         |
| .24         | Speed limit (30 km/h)        |
| .23         | Priority road                |
| .05         | Yield                        |
| .05         | Speed limit (80 km/h)        |

Third sign is the first one with error. It is predicted as "Dangerous curve to the left" with low certainty (0.43 probability) but the distance from that prediction to the correct one is not very large (13 percentage points), with the correct prediction in second place.

| Probability |     Prediction               | 
|:-----------:|:----------------------------:| 
| .43         | Dangerous curve to the left  |
| .30         | Slippery Road                |
| .09         | Bicycles crossing            |
| .08         | Wild animals crossing        |
| .07         | Double curve                 |

The fourth prediction is one I had a lot of trouble with was the fourth traffic sign, for which the correct class didn't even make the top 5. The correct classification is: "double curve". The top 5 were "Beware of ice/snow", "children crossing", "dangerous curve to the right", "slippery road" and "bicycles crossing". All the provided predictions correspond to signs enclosed in a warning triangle, but they are all obviously incorrect.

| Probability |     Prediction               | 
|:-----------:|:----------------------------:| 
| .40         | Beware of ice/snow           |
| .36         | Children crossing            |
| .08         | Dangerous curve to the right |
| .07         | Slippery road                |
| .06         | Bicycles crossing            |

The fifth example is also misclassified, with 0.64 probability that it is a stop sign and only 0.28 to the correct class of "No entry".

| Probability |     Prediction                    | 
|:-----------:|:---------------------------------:| 
| .64         | Stop                              |
| .28         | No entry                          |
| .03         | Vehicles over 3.5 tons prohibited |
| .02         | Dangerous curve to the right      |
| .01         | Wild animals crossing             |

The sixth image has the model pretty sure that it is a 30 km/h limit sign with 99% certainty. Speed limit signs in general seem very recognizable.

| Probability |     Prediction               | 
|:-----------:|:----------------------------:| 
| .99         | Speed limit (30 km/h)        |
| .00         | Speed limit (50 km/h)        |
| .00         | Speed limit (80 km/h)        |
| .00         | Speed limit (70 km/h)        |
| .00         | Speed limit (60 km/h)        |

The seventh sign is likewise very specifically labelled "Yield" with 100% accuracy. This one is pretty recognizable being the only one that is an inverted triangle.

| Probability |     Prediction               | 
|:-----------:|:----------------------------:| 
| 1.0         | Yield                        |
| .00         | Road works                   |
| .00         | Ahead only                   |
| .00         | Priority road                |
| .00         | No vehicles                  |

The eighth sign has a high accuracy as well (0.86 probability and 80 percentage points away from the closest match) and is classified correctly as "Children crossing".

| Probability |     Prediction                        | 
|:-----------:|:-------------------------------------:| 
| .86         | Children crossing                     |
| .06         | Right of way in the next intersection |
| .04         | Beware of ice/snow                    |
| .02         | Road narrows on the right             |
| .00         | Bicycles crossing                     |

The last sign was also identified very strongly. Some of the triangular signs with bigger blobs of black are easily recognizable because their position varies inside the triangle's frame.

| Probability |     Prediction               | 
|:-----------:|:----------------------------:| 
| .99         | Road works                   |
| .00         | Bumpy road                   |
| .00         | General caution              |
| .00         | Bicycles crossing            |
| .00         | Beware of ice/snow           |


I think that more detailed features inside the signs present problems while the bigger sign shape features are guessed better. Possibly smaller filters in the convolutions would be able to catch the differences better, and again allowing the model to train for longer.

It should be possible to infer that if the top N predictions don't present a sufficiently large difference, that the model is unsure and a human should assist the decision.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I plotted the first three convolution featuremaps and what I can see is the first layer being activated by upwards slanting lines, and some of the content of the sign and downward lines to lesser degree.

The second layer was more interested in the information in and around the sign, but showed less intensity in the activations although more activity overall.

The third layer, which already presents some data compression shows disjointed activations although there are some indications of the sign borders, especially on the upwards slanting line of the sign's border which happens to be the side catching the light and thus more easily differentiable than the other side. Little to no activation is displayed matching the shape in the center of the sign. Since the example is similar to the sign that was misclassified most, it is telling that there are no such activations.
