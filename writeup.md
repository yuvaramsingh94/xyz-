## Project: Follow me


---


[//]: # (Image References)

[image1]: ./model.png

## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Network architecture
#### 1. Clearly explains each layer of the network architecture and the role that it plays in the overall network.
A three-layer encoder/decoder setup is used in this project. The encoder blocks are comprised of a separable ConvNet and a batch normalization step. Each encoder layer uses kernel size of 3 and stride of 2, yielding a half-sized width/height for the next layer. The filter sizes are set up in ascending order. An 1x1 ConvNet is added after the encoder steps. Each decoder block includes a concatenated large and small layer. The small layer, which is the later layer of the encoder blocks, is upsampled by a factor of 2, which reverses the halving of the encoder outputs. Two separable layers with stride of 1 are added in the decoder block. The complete structure is generated using Keras plot as included. In the end, there is a fully connected layer to get the final output.

![alt text][image1]

#### 2. Explain the neural network parameters including the values selected and how these values were obtained (i.e. how was hyper tuning performed? Brute force, etc.)
Larger batch size yields better results, but also takes longer to compute. I found a batch size of 32 is a reasonable choice. Initially, a small learning rate of 0.001 was used, but the model quickly fell into overfitting within 3 epochs. Reflected in the testing, a great more false positives in the no-target scenario and false negatives in the target-far scenario were observed. A learning rate of 0.01 reduced the overfitting problem, but more epochs were needed as the validation loss curve converged slower. I initially used 8 epochs but found that increasing it to 20 would yield much better results.


#### 3. Demonstrate a clear understanding of 1 by 1 convolutions and a fully connected layer and where/when/how they should be used
The 1x1 convolution layer convolve through the depth of the previous layer pixel by pixel, thus it can be a way for dimension reduction by reducing the filter size. In addition, it is a way to increase network depth (as each layer is accompanied with a nonlinear activation like relu) and hence improve performance.

A fully connected layer is attached to the end of the network. It converts each pixel of the outputs from the previous layer to the the dimension the same as the number of classes. The probabilities are determined through the softmax function.

#### 4. Identify the use of various reasons for encoding / decoding images, when it should be used, why it is useful, and any problems that may arise.

The encoder takes the image and converts it to a high-depth feature volume. While it extracts abstract features, it loses spatial information. The decoder images gradually add back the previous layers through convoluting a concatenated upsampled deeper layer with a superficial layer, and consequently recover the spatial information as well as classify the pixels by the extracted features.

#### 5. Whether this model and data would work well for following another object (dog, cat, car, etc.) instead of a human and if not, what changes would be required.

This data would probably not work well for following the objects that never show up in the training data. A set of train/validation data containing those objects are needed. The model should allow more number of classes. However, since the existing model already extracted basic features, the new training may only need to train the layers close to the output.

#### 6. Model
The performance of the model yields 0.52 final grade score. Overall, it performed quite well for the images taken following behind the target, relatively good for the images without the target, but failed to detect more than 1/3 of the target in the far-away scenario. This may be solved by increasing the number of epochs and the learning depth.
