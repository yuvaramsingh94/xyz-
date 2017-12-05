## Project: Follow Me

---

**The goals / steps of this project are the following:**

* Clone the project repo here
* Fill out the TODO's in the project code as mentioned here
* Optimize your network and hyper-parameters.
* Train your network and achieve an accuracy of 40% (0.40) using the Intersection over Union IoU metric which is * final_grade_score at the bottom of your notebook.
* Make a brief writeup report summarizing why you made the choices you did in building the network.

[//]: # (Image References)

[image1]: ./image1.jpg
[image2]: ./image2.jpg


## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

#### 2. Network architecture

For this project I used a Fully-convolutional Neural Network (FCN), which allows us to do semantic segmentation and pinpoint exactly where a certain object is on an image. The network reads in images as inputs and assigns weights to each pixel in its output layer, allowing us to classify the object in any location. Unlike CNNs, which contain fully-connected layers and hence lose spatial information, FNNs use only convolutional layers.

The overall structure of the network I built consists of two main parts - an encoding block and a decoding block, which are connected by a 1x1 convolutional layer. The encoding block extracts features used for segmentation, starting with layers that recognize simpler shapes and patterns and then proceeding with deeper layers that can learn more complex shapes. The 1x1 layer has roughly the same function and it fully maintains spatial information. The following decoding layers that are connected to it gradually decrease in depth and increase in size, ultimately restoring the dimensions of the input image. There are also skip connections along the way, whose purpose is to skip some intermediate layers and connect directly to non-adjacent layers. Thus they can preserve some information that might otherwise be lost in the encoding. The final output layer after the decoding block is a convolutional layer with softmax activation, which yields the classification between the three classes for every pixel.

After experimenting with a couple of different models, I decided on an architecture that uses two encoder layers, one 1x1 convolution layer, and two decoder layers. The filter sizes for the encoder layers are 32 and 64, respectively, with the convolution layer having a filter size of 128, while the decoder layer reduce back the size from 64 to 32, respectively. The stride of two and the same padding of the encoding and decoding layers have the overall effect of halving the image size while increasing depth.

layer | filter size | stride
--- | --- | ---
encoder 1 | 32 | 2
encoder 2 | 64 | 2
1x1 convolution | 128 | 1
decoder 1 | 64 | 2
decoder 2 | 32 | 2

The output of the last decoder layer is connected to an output convolution with softmax activation.

![alt text][image1]

A couple of alternative designs I tried:

  - I added an extra convolutional and decoder layers to increase depth. This was making training too slow and not giving any accuracy improvements.
  - I tried using larger filter sizes: 64 -> 128 -> 256 -> 128 -> 64. Training was slower and the score wasn't improving significantly, possibly due to requiring more epoch or tweaks in the other parameters.

Ultimately, the architecture with the best score was the one outlined above, and it also had the benefit of having a faster training time, which allowed me to iterate faster in the tuning of the network hyper parameters.

#### 3. Network parameters

Here are the final network parameters that I used:

    learning_rate = 0.002
    batch_size = 32
    num_epochs = 100
    steps_per_epoch = 200
    validation_steps = 50
    workers = 2

I immediately started using the AWS p2 GPU instance and I never tried training locally on my laptop, which I suspect would've been extremely slow.

First, I ran a couple of experiments with the number of workers (`workers`) parameter keeping all values set to the default except for `num_epoch = 1`. It seemed the default value of `2` was the best option even for the AWS GPU instance. I didn't get a noticeable improvement with `4` and performance was degrading with more workers.

Regarding, `validation_steps`, `steps_per_epoch`, and `num_epochs` parameters, I was guided by a few simple rules that I thought made sense to follow:

- The training set contained `5350` images (since I had added some extra images) and the validation set had `1184` images. Hence, it made sense to maintain the same proportion for `steps_per_epoch` relative to `validation_steps`, i.e. `steps_per_epoch = 4.5 * validation_steps`.
- The rough rule that `steps_per_epoch` should be some multiple of the total number of images in the training dataset divided by the `batch_size`. This is to make sure that the vast majority of the training set is "seen" by the training algorithm.

I was planning to start training with a default of `10` epochs and use the epoch number as a means to control overfitting. This means that roughly `500` images should be covered per epoch (preferable some larger multiple of that). Since batches are random and I wanted to get good coverage of the training set, I started with a batch size of `25`. Later, I experimented with batch sizes of `50` and reduced `steps_per_epoch` to `110` and `validation_steps` to `25`. These values mean that `steps_per_epoch * num_epochs * batch_size = 5500` images will be processed per epoch. I also tried batch size of `100` but this lead to instability in the training loss and validation loss, possibly indicating overfitting and the model not being able to generalize (score `0.3189`). The final value I got the best results with was using a batch size of `32`, and restoring the number of `steps_per_epoch` to `200` so that I have good coverage of the training set.

I set the `learning_rate` through experimentation. I started with a `learning_rate` of `0.01` (score `0.3959`) and proceeded to reduce it to `0.001` and then slightly up to `0.002` as a safe enough value that would allow me to decrease the number of epochs. This was also the value that gave me the best IoU scores. The previous high learning rate was causing the validation loss to fluctuate a lot, possibly due to noise.

![alt text][image2]

Ultimately I was able to obtain a score of `0.442620757002`.

#### 4. Techniques and concepts in the neural network layers (1x1 vs fully connected layer)

The most notable technique in the FCN is the use of a 1x1 convolution layer, which allows the network to retain spatial information from the encoding block. Unlike fully connected layers where the data is flattened to two dimensions and spatial information is lost, 1x1 convolution increases depth and maintains volume of the previous layer so that location information can be preserved.

#### 5. Image manipulation concepts

Another notable feature of FCNs is the use of encoding and decoding blocks. An encoding block reads in an image and effectively recognizes specific features. Later, these feature maps are up-scaled in the decoding layer up to the original image dimension, allowing classification to be carried out at the pixel level. Skip connections also help preserve certain low-level information like pixel colors or edge information that might get lost in subsequent encoding layers.

#### 6. Network limitations

Currently, the network has only three output dimensions and hence three output classes. As such it can't be used to classify more than 3 classes, like 'cat', 'dog', etc. To do that, we would need to change the depth.

Additionally, the network would need to be partially retrained to recognize a different type of object. This can be achieved by taking out the encoding block and replacing it with another encoding block that was trained for a different object. The encoder/decoder structure and the use of skip connections means that the network would need to be retrained at that point.


### Model

#### 1. The model is submitted in the correct format.

The enclosed `config_model_weights` and `model_weights` contain the model saved in the `.h5` format.

#### 2. The neural network must achieve a minimum level of accuracy for the network implemented.

The level of accuracy achieved with the enclosed model is `0.442620757002`.

### Future Enhancements

- Gather better data:
  - It seems most of the images don't contain the hero, and the training set could be improved if the proportion of images with the hero was higher.
  - Most of the low scores were due to many false negatives when the hero is far away. Potentially, gather more images of the hero in the distance.
- Better hyper parameter search.
