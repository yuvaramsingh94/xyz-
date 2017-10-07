

[//]: # (Image References)

[image1]: ./img/evaluation.png "evaluation"
[image2]: ./img/final_score.png "score"
[image3]: ./img/prediction_1.png "pred 1"
[image4]: ./img/prediction_2.png "pred 2"
[image5]: ./img/main_model_with_shapes.png "model"

![alt text][image4]

This is the last project of udacity's Robotics Nanodegree term 1, where we get to play wiht Neural network concepts like Deep neural networks ,convolutional neural networks and semantic segmentation . this project implements semantic segmmentation to train a model for detecting and locating our hero in a croweded city environment and make our quad follow her . through this project we can see the practical implimentation of semantic segmentation  


## Rubrics explination 

### Network architecture 
<image of the network graph>
 
 ![alt text][image5]
 


#### Encoder block

each Encoder block consist of seperable convolution layer fitted with batch normalizar which normalize the output of one layer and sends that to the next layer . this normalization helps in creating sub neural net and improves the training time . also it allowes for hight learning rate which will come in handy if you try to train the model for higher epoch . seperable conv2D is used to reduce the total number of parameters to train the system . 

#### Decoder block

Decoder block consist of bilinear up-sampling layer , concatenation and a seperable conv2d layer fitted wiht batch normalization . the up-sampling layer uses weighter resampling technique to increase the dimension of the input .

P12 = P1 + W1*(P2 - P1)/W2  

this is the formula by which the up sampler increases the dimentions 

after upsampling , the result is concatenated with a layer from encoder which has the same dimention . this technique is called skip connection which helps in retaining the useful information . a seperable conv2D layer is added to the end of the skipconnection . this conv2d layer uses stride value of 1 so there is not change in the dimentionality 

#### 1x1 convolution layer 

1x1 conv is a way of retaining the spacial information which will be loosed if we use fully connected layer . fundamentally it is same as the convolution 2d layer with same padding , stride of 1 and 1x1 kernal .  this provides an advantage of using different size image as the input since we dont have a fixed number of nuerons as teh fully connected layer

#### final architecture

as you can see in the model.summery() output provided above , i have used 32,64,128 as filter size for the encoder and reconstructed them using the same filter size and used three skip connections . final layer output shape is same as the input provided (160,160,3). 

### Hyper parameter 

###### my parameters used 

* Epoch = 30
* Batch size = 16 (can be changed )
* steps per epoch = 280 (i used 4100+ training data )
* validation_steps = 50
* workers = 2
* learning rate = 0.01
* optimizer = Adam optimizer
* passes per batch = 30 X 280 = 8400

i trainied my model over 4100+training data and 1500+ validation data . this forced me to run the training session for a more epochs . i setteled on 30 epochs because it gave me a good ourput and after that the validation loss started to oscillate up and down . my GPU suffered from low memory and this is the reason i choose a low batch size 16 so i can train my model without crashing my system . this can always be changed based on the hardware . to speed up the convergence process , i choose a learning rate of 0.01 after trying our several rates like 0.001,0.005 etc . this rate seems to work good for me on the training data i collected 

### limitations 

* my detections have black spots in it due to the lack of activation . this can be avoided by training the model for few more epoch and also increasing the dataset can help this 
* A descent GPU is required to run a inference of my model . this makes it hard to impliment this model in a small drone which should relay on a GPU server to process the live feeds
* The model is trained only on 3 labels so the pixel classification is limited to those 3 labels . to use this model for a much wider classification problem , retraining or transfer learning is required
* few miss clasification of environment pixels can be seen . the main cause for this is the similarities on the color of hte building and the people . this can also be avoided by using more data 

### Output of the model


![alt text][image1]
![alt text][image2]
![alt text][image3]


 
