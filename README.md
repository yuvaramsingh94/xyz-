# Follow me project

## About Project Follow me

This is the last project of udacity's Robotics Nanodegree term 1, where we get to play wiht Neural network concepts like Deep neural networks ,convolutional neural networks and semantic segmentation . this project implements semantic segmmentation to train a model for detecting and locating our hero in a croweded city environment and make our quad follow her . through this project we can see the practical implimentation of semantic segmentation  


## Rubrics explination 

### Network architecture 
<image of the network graph>
 ''' 
 
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
 
 '''


 
