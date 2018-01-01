# Follow Me Writeup
The following discusses an implementation of a fully convolutional neural network to perform semantic segmentation on an image stream in order to enable specified target following by a ‘quadcopter’ within a simulated environment.
## Problem Overview
The central problem this project addresses is semantic segmentation of an image. In our scenario, we have a quadcopter with a camera that we would like to follow a specific person. A dataset of example images has been provided as well as ground truth segmentation of these images. The proposed solution implements supervised learning to train a fully convolutional neural network to perform semantic segmentation on images to classify our target person, other people, and the background. The effectiveness of the solution is measured using Intersection over Union of the segmentation compared to the ground truth.

## Network Architecture
As stated, the model implemented is a fully convolutional neural network. The network consists of 5 layers, 2 encoders, a 1x1 convolution, and 2 decoder layers. Encoding enables us to extract information from the images through the use of learned filters. Decoding then enables to apply this knowledge to segment the image by classify each pixel into one of the trained categories. Care must be taken when using this architecture to avoid overfitting due to the large number of parameters that are being tuned as well as ensuring the layers are structured properly to extract the information needed. Two encoder layers are implemented using separable convolutions and batch normalization and ReLU activation functions to create the nonlinear model with the following ```encoder_block```. Separable convolutions are used over regular convolutions as they require fewer parameters and improve efficiency of the network. The batch normalization allows us to normalize inputs between layers to improve training much as normalizing the original inputs to the network.

```python
def encoder_block(input_layer, filters, strides):  
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer
```


Which used the further helper functions ```separable_conv2d_batchnorm```
```python
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)

    output_layer = layers.BatchNormalization()(output_layer)
    return output_layer
```
and ```conv2d_batchnorm```
```python
def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                      padding='same', activation='relu')(input_layer)

    output_layer = layers.BatchNormalization()(output_layer)
    return output_layer
```

After the two encoder layers is a 1x1 convolution. This is used to reduce the dimensionality of network parameters compared to a fully connected layer while retaining spatial information from the initial encoding. This retention of spatial information allows for pixelwise segementation versus image classification. This is implemented with a regular convolution with batch normalization using the conv2d_batchnorm helper function shown previously.
Following the 1x1 convolution are two decoder layers. These layers consist of a bilinear upsampling by a factor of 2 to increase the spatial dimensions through interpolation of the output to prepare to concatenate layers. The next step concatenates the upsampled layer with a skip layer from earlier in the network. This is used to retain information that may have been lost otherwise due to the encoding process. After this are two separable convolutions to extract information from the concatenated layers. The basic structure of the decoder block is shown below.

```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    upsampled_layer = bilinear_upsample(small_ip_layer)
    print('upsampled: ',upsampled_layer.shape)
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    concatenated_layer = layers.concatenate([upsampled_layer, large_ip_layer])
    print('concate: ',concatenated_layer.shape)
    # TODO Add some number of separable convolution layers
    sep_conv_1 = separable_conv2d_batchnorm(concatenated_layer, filters, strides=1)
    output_layer = separable_conv2d_batchnorm(sep_conv_1, filters, strides=1)
    return output_layer    
```

After building up the individual elements, we can define the structure of the complete network. The parameters that define the network are shown below. The output layer is a fully connected layer with a softmax activation. This enables the network to classify each individual pixel within the image.

```python
def fcn_model(inputs, num_classes):
    print('Input: ',inputs.shape)
    # TODO Add Encoder Blocks.
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encoder_1 = encoder_block(inputs, filters=32, strides=2)
    print('Encoder 1: ',encoder_1.shape)
    encoder_2 = encoder_block(encoder_1, filters=64, strides=2)
    print('Encoder 2: ',encoder_2.shape)
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv = conv2d_batchnorm(encoder_2,filters=32,kernel_size=1,strides=1)
    print('Conv: ',conv.shape)
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder_1 = decoder_block(conv, encoder_1, filters=64)
    print('Decoder 1: ',decoder_1.shape)
    x = decoder_block(decoder_1, inputs, filters=32)
    print('Decoder 2: ',x.shape)
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```

![Network Diagram](Network_Diagram.jpg)

Following the construction of our model, attention is focused on tuning the require hyperparameters to reach the goal of semantic segmentation. Below are the hyperparameters used which were chosen using an experimental methodology.

```python
learning_rate = 0.002
batch_size = 128
num_epochs = 40
steps_per_epoch = 50
validation_steps = 15
workers = 2
```
## Model Parameters

#### batch_size:
The number of training samples/images that get propagated through the 	network in a single pass. As we are using batch normalization, larger batch sizes will ensure this normalization is as effective as possible. The upper limit is usually determined by memory available for storing the model.

#### num_epochs:
The number of times the entire training dataset gets propagated through the network. This value was determined by observing the graph of loss vs training time. One obvious upper limit is when training and validation losses start to diverge, indicating the network has overfit the training data and fails to generalize.

#### steps_per_epoch:
The number of batches of training images that go through the network in 1 epoch. This value was based on the total number of images in training dataset divided by the batch_size and increased to the current value.

#### validation_steps:
The number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset. This was chosen based on the number of validation images divided by the batch set to ensure each image was processed through the network in each epoch and increased during testing.

#### workers:
The maximum number of processes to spin up. This can affect your training speed and is dependent on your hardware.



## Results Discussion
After training the model, the final evaluation score was 40.9%. This was calculated using a weighted intersection over union indicates the average success rate of the model in predicting the classification of a given pixel correctly under three scenarios:

    patrol_with_targ: Test how well the network can detect the hero from a distance.
    patrol_non_targ: Test how often the network makes a mistake and identifies the wrong person as the target.
    following_images: Test how well the network can identify the target while following them.

As currently trained, this model would not be very useful for other target following tasks unless it was tweaked with further training image datasets. The basic structure and features it has 'learned' can be more broadly applied but require further training to perform other tasks in this domain successfully.
## Future Enhancements
The experimental method for choosing hyperparameters is a notably inefficient method and the subject of much current research. One future enhancement would include tuning the hyperparameters by using Bayesian search to create a full learning model.  Other improvements would involve expanding the available dataset for training as the base dataset used to create this model included a rather small 4131 images.
