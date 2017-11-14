[arch]: ./images/arch.jpg

#### Project: Follow me
---

##### The write-up conveys the an understanding of the network architecture.

The student clearly explains each layer of the network architecture and the role that it plays in the overall network. The student can demonstrate the benefits and/or drawbacks different network architectures pertaining to this project and can justify the current network with factual data. Any choice of configurable parameters should also be explained in the network architecture.

The student shall also provide a graph, table, diagram, illustration or figure for the overall network to serve as a reference for the reviewer.

---

Since what the network should output is an image with each pixel representing the likelihood of it being part of the target, I used a fully convolutional network, with encoders and decoders and a normalizing layer in-between.

For the decoders to be able to better reconstruct the output image from the 1x1 convolutional layer, I added skip connections to the decoding layers. The network is very similar to the one we had to build for the last lab before the assignment.

The number of encoders and decoders is 3. The kernel size for the convolution layers is 3. The size of the 1x1 layer I went with was 256. It was enough to get a score above 0.40, but most likely the network can be improved by deepening it. However, given the fact that the input image is quite small (160x160), I don't think a whole lot more can be done since it becomes hard even for a human to tell people apart in an image that size.

![architecture][arch]

---

##### The write-up conveys the student's understanding of the parameters chosen for the the neural network.

The student explains their neural network parameters including the values selected and how these values were obtained (i.e. how was hyper tuning performed? Brute force, etc.)
All configurable parameters should be explicitly stated and justified.

---

The parameters for the neural network were the following:

The *learning rate* provides a way to tune how fast the neural net adjusts to fit new samples that are used for training. The smaller the learning rate, the more times it has to be shown the training samples in order to be able to improve to fit them correctly.

The *number of epochs* is the number of times the network is shown the full set of training samples. The bigger the number of epochs, the smaller the error should get. However, training a network for a large number of epochs could make it overfit the training set, leading to a worse performance normally on other samples.

The training samples are split into batches. Then, for each one of the batches, the training samples are passed throught the network and the weights are adjusted. The *batch size* represents the number of samples that are in a batch. The larger the number of samples in a batch, the more precise the loss estimation can be. However, having a lot of samples in a batch makes training slower.

*Steps per epoch* represents the number of batches in the training set.

*Validation steps* represents the number of batches in the validation set.

By changing the number of *workers*, you could use more cores in the training, probably because the computation runs on the GPUs, this one does not influence how fast the network trains.

The values for the parameters:

```
learning_rate = 0.0016
batch_size = 35
num_epochs = 25
steps_per_epoch = 152
validation_steps = 30
workers = 4
```

To get to these values, I started with a set of values (0.001 for learning rate, 50 for batch size, 20 epochs) and then changed them until I got a passing score. Unfortunately, I realized at some point that on most of my tweaking runs, I was continuing from the same model (by re-running the training cell) instead of re-running the whole notebook which reinitializes the model weights. :(

When doing the search, I was looking for a less steep decline in loss in the first three epochs and for a decent training speed since I did a lot of runs -- almost used all my $100 credit, most likely wasted more than half of it unfortunately.

I recorded extra images for the training, so in the end I had 5244 samples for training, hence the value for steps_per_epoch, to be able to cover all samples every epoch.

---

##### The student has a clear understanding and is able to identify the use of various techniques and concepts in network layers indicated by the write-up.

The student is demonstrates a clear understanding of 1 by 1 convolutions and where/when/how it should be used.
The student demonstrates a clear understanding of a fully connected layer and where/when/how it should be used.

---

1x1 convolutions could be used for dimensionality reduction, since they summarize pixel data from multiple channels to a smaller number of new channels. A 1x1 convolution is used in the last encoding part of the FCN. The number of filters is the new number of channels that the 1x1 layers summarizes for one pixel of the image.

Because the data comes from the channels of the sample pixel, and because the convolution is followed by a batch normalization with a ReLUs, the 1x1 convolution also becomes a non-linear transformer of the channels in the previous layer.

However, this reduction in dimension takes place at the channel/filter level, not at the spatial level (width or height).

In this particular architecture, a 1x1 convolution is used in the middle layer to add non-linearity to the model.

In fully connected layers, all outputs are connected to all the activations in the previous layer. These layers are especially good at classification, since for a pixel that's classified, channels of the pixels that are near it are also considered when training.

---

##### The student has a clear understanding of image manipulation in the context of the project indicated by the write-up.

The student is able to identify the use of various reasons for encoding / decoding images, when it should be used, why it is useful, and any problems that may arise.

---

There are two parts to the neural network I used in the project: an encoding part and a decoding part that trained together, end-to-end.

At each one of the three layers of the encoding part, the actual height and width of the output become smaller, so we can talk about a reduction in the size of the image on these two dimensions. However, the network makes up for this loss in height and width by creating a deeper model of the image (can be seen of the diagram). After the last encoding layer, the image is transformed to a 20 x 20 x 256.

In the decoding layers, bilinear interpolation is used to upscale the image coming from the previous layer to an image of the same size (height and width) as the output of that layer. That image is also concatenated to the image of the same size coming from the corresponding encoding layer.

Being trained end-to-end, the network improves both the encoding half (how well it can summarize what's going on in the original training sample), as well as the decoding half (how well it can reconstruct the position of the detected objects from the data encoded in the middle layer).

In the context of image manipulation (if you view the neural network as a photo filter), the network takes as input a photo taken by the drone camera and produces a photo of the same size, in which each pixel is colored depending what class is the object that the neural network thinks that pixel is part of. Hence, the detected objects lose their detailed features, being replaced by blobs of color, each class having its own color.

---

##### The student displays a solid understanding of the limitations to the neural network with the given data chosen for various follow-me scenarios which are conveyed in the write-up.

The student is able to clearly articulate whether this model and data would work well for following another object (dog, cat, car, etc.) instead of a human and if not, what changes would be required.

---

This model is obviously trained for detecting the person wearing red, but with different training data, the same architecture most likely can be trained to detect other types of objects.

---

##### Future improvements

Camera resolution could be improved in order to provide a more detailed view of what's in front of the drone.
More samples could be recorded for both training and validation sets in order to improve the accuracy of the model.
More layers could be added for an even deeper encoding of the given samples.

Of course, none of these will work without also re-training the neural network.
Running the training for more epochs would also improve the accuracy of the model.
