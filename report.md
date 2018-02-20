>>>>>>>>>>>>>> Report for Follow Me project <<<<<<<<<<<<<<<<<<
Overview
In this project I used a fully-convolutional neural network for pixel level classification to identify "hero" from other people. The data set I used is from Udacity's default data. The model was trained using Amazon Web Services (AWS) instance.

Network architecture
The architecture is an encoder-decoder architecture of a Fully-Convolutional Neural Network. I used two layers for each encoder and decoder part, which achieves the project requirement. The model consists of two encoder and two decoders and 1X1 convolution layer between them. The first layer uses filter size 32 with a stride of 2 and second layer uses filter size 64 with a stride of 2.
Encoder: 
The encoder layer builds a depth of understanding specific features of the segmentation, each successive layers builds different aspects of the object. For example, the first layer can learn to distinguish basic characteristics such as lines, brightness and next layer can identify more complicated shapes which contains a conbination of simple shapes.
1X1 Convolution:
1X1 Convolution layer is used to connect encoder and decoder layers, the convolution layer helps to retain spatial information. Besides 1X1 convolution layer can feed in image of any size compared with the limitation of fully connected layers.
Decoder:
The decoder section can be composed of transposed convolution layers or bilinear upsampling layers. Each decoder is able to reconstruct a little more spatial information. The final decoder layer will output a layer the same size as the original image.
Skip connection:
The skip connection helps to retain information which was lost in previous subsequent layers to achieve more precise robust segmentation.

Parameters chosen
learning rate -> 0.0015
batch size -> 40
epochs -> 40
steps_per_epoch -> 200
validation_steps -> 50
workers -> 2
I found high learning rate often overshoot desired ouput and finally 0.0015 works for me. Computing over entire data set is expensive so I chose 40 as batch size. Other parameters were found mostly based on try-and-error. 

Discussion
The model was only trained to identify personel in an image, it could be used for identifying other objects (car, dog, cat, etc) with similar complexity. The data needs to be recollected and model might needs to be changed (e.g. add more layers) for more complex objects.

Future enhancement
I noticed some of the classified images miss some pixels belongs to the personel, which could because of the brightness is similar to the background. This might be improved if we choose another color channel. And some people far away are not classified in good quality, this could be improved with more data (I just used the default dataset) and with deeper layers

Model weight file and other files are attached in the same folder, also the architecture diagram

The final score of the model is 0.419
