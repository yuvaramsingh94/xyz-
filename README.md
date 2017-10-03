

This is the last project of udacity's Robotics Nanodegree term 1, where we get to play wiht Neural network concepts like Deep neural networks ,convolutional neural networks and semantic segmentation . this project implements semantic segmmentation to train a model for detecting and locating our hero in a croweded city environment and make our quad follow her . through this project we can see the practical implimentation of semantic segmentation  


## Rubrics explination 

### Network architecture 
<image of the network graph>
 
 
 
 Layer (type)                 Output Shape              Param #   
=================================================================
input_3 (InputLayer)         (None, 160, 160, 3)       0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 160, 160, 32)      896       
_________________________________________________________________
batch_normalization_23 (Batc (None, 160, 160, 32)      128       
_________________________________________________________________
separable_conv2d_keras_19 (S (None, 80, 80, 32)        1344      
_________________________________________________________________
batch_normalization_24 (Batc (None, 80, 80, 32)        128       
_________________________________________________________________
separable_conv2d_keras_20 (S (None, 40, 40, 64)        2400      
_________________________________________________________________
batch_normalization_25 (Batc (None, 40, 40, 64)        256       
_________________________________________________________________
separable_conv2d_keras_21 (S (None, 20, 20, 128)       8896      
_________________________________________________________________
batch_normalization_26 (Batc (None, 20, 20, 128)       512       
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 20, 20, 128)       16512     
_________________________________________________________________
batch_normalization_27 (Batc (None, 20, 20, 128)       512       
_________________________________________________________________
bilinear_up_sampling2d_10 (B (None, 40, 40, 128)       0         
_________________________________________________________________
concatenate_10 (Concatenate) (None, 40, 40, 192)       0         
_________________________________________________________________
separable_conv2d_keras_22 (S (None, 40, 40, 128)       26432     
_________________________________________________________________
batch_normalization_28 (Batc (None, 40, 40, 128)       512       
_________________________________________________________________
bilinear_up_sampling2d_11 (B (None, 80, 80, 128)       0         
_________________________________________________________________
concatenate_11 (Concatenate) (None, 80, 80, 160)       0         
_________________________________________________________________
separable_conv2d_keras_23 (S (None, 80, 80, 64)        11744     
_________________________________________________________________
batch_normalization_29 (Batc (None, 80, 80, 64)        256       
_________________________________________________________________
bilinear_up_sampling2d_12 (B (None, 160, 160, 64)      0         
_________________________________________________________________
concatenate_12 (Concatenate) (None, 160, 160, 96)      0         
_________________________________________________________________
separable_conv2d_keras_24 (S (None, 160, 160, 32)      3968      
_________________________________________________________________
batch_normalization_30 (Batc (None, 160, 160, 32)      128       
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 160, 160, 3)       867       
=================================================================
Total params: 75,491
Trainable params: 74,275
Non-trainable params: 1,216
 

#### Encoder block

each Encoder block consist of seperable convolution layer fitted with batch normalizar which normalize the output of one layer and sends that to the next layer . this normalization helps in creating sub neural net and improves the training time . also it allowes for hight learning rate which will come in handy if you try to train the model for higher epoch . seperable conv2D is used to reduce the total number of parameters to train the system . 

#### Decoder block

Decoder block consist of bilinear up-sampling layer , concatenation and a seperable conv2d layer fitted wiht batch normalization . the up-sampling layer uses weighter resampling technique to increase the dimension of the input .

P12 = P1 + W1*(P2 - P1)/W2  

this is the formula by which the up sampler increases the dimentions 

after upsampling , the result is concatenated with a layer from encoder which has the same dimention . this technique is called skip connection which helps in retaining the useful information . a seperable conv2D layer is added to the end of the skipconnection . this conv2d layer uses stride value of 1 so there is not change in the dimentionality 

#### 1x1 convolution layer 

1x1 conv is a way of retaining the spacial information which will be loosed if we use fully connected layer . this is 

#### final architecture

as you can see in the model.summery() output provided above , i have used 32,64,128 as filter size for the encoder and reconstructed them using the same filter size and used three skip connections . final layer output shape is same as the input provided (160,160,3). 

### Hyper parameter 

######my parameters used 

* Epoch = 30
* Batch size = 16 (can be changed )
* steps per epoch = 280 (i used 4100+ training data )
* validation_steps = 50
* workers = 2
* learning rate = 0.01
* optimizer = Adam optimizer

i trainied my model over 4100+training data and 1500+ validation data . this forced me to run the training session for a more epochs . i setteled on 30 epochs because it gave me a good ourput and after that the validation loss started to oscillate up and down . my GPU suffered from low memory and this is the reason i choose a low batch size 16 so i can train my model without crashing my system . this can always be changed based on the hardware . to speed up the convergence process , i choose a learning rate of 0.01 after trying our seeral rates like 0.001,0.005 etc . this rate seems to work good for me at the training data i collected 

### limitations 

* A descent GPU is required to run a inference of my model . this is the current bottle neck 





 
