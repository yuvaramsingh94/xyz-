# Project: Follow Me

---

## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points

Note: I have been helped through the Udacity Slack.

---

## Introduction
The Follow Me project consists in making a UAV able to follow a *specific* human.
To achieve this goal, we build and train a fully convolutional network (FCN).

## Fully Convolutional Network (FCN)

### Encoders
First, the RGB image is taken to encoders.
```python
encoder1 = encoder_block(inputs, filters=128, strides=2)
encoder2 = encoder_block(encoder1, filters=64, strides=2)
encoder3  = encoder_block(encoder2, filters=32, strides=2)
```

The encoder role is to extract features from the image.
The filters number is the depth of the output of each encoder (The input RGB image, having three colors channel, is of depth 3).
The strides number is the dimension divider of the encoder. With strides=2, the encoder output dimensions will be half of the encoder input dimensions.

### Decoders
Symmetrically, to the encoders, we put decoders.
```python
decoder3 = decoder_block(convolution1x1, large_ip_layer=encoder2, filters=32)
decoder2 = decoder_block(decoder3, large_ip_layer=encoder1, filters=64)
decoder1  = decoder_block(decoder2, large_ip_layer=inputs, filters=128)
```

The decoder role is to upscale the output of encoders (features) to the same size as the original image. That is why there is one decoder linked to each encoder, mirroring the same filter parameter.

### 1x1 Convolution layer
As explained in the course, we put a 1x1 convolution layer between the encoders and decoders.
The idea is to increase the depth by adding this "mini neural network" working over the patch. The good thing is, being a 1x1 convolution layer, the computing cost is low (mathematically, it is matrix multiplications).
Therefore, it offers more deepness for a cheap computive cost.

### Mixed up filter depth
The course lecture video on Fully Convolutional Network (https://classroom.udacity.com/nanodegrees/nd209/parts/c199593e-1e9a-4830-8e29-2c86f70f489e/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/032b020f-7c02-46dc-9266-eaee3eb76eb7/concepts/e5e4584d-c31a-4a1c-aa36-1ff06c608956#) shows, in that order:
- decreasing depth convolutional layers
- increasing depth convolutional layers
- the global resulting scheme

This order in explanation initially misled me in thinking that the first part (therefore encoder) was the decreasing depth, and the lst part (therefore decoder) was the increasing depth, whereas it is the opposite.

Fortunately, it still provided good results; although re-ordering the depth and training on an expensive cloud again might led to better results at the end

### Skip connections
As explained in the skip connections video course (https://classroom.udacity.com/nanodegrees/nd209/parts/c199593e-1e9a-4830-8e29-2c86f70f489e/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/032b020f-7c02-46dc-9266-eaee3eb76eb7/concepts/30527098-b8b0-419a-821b-da4473f39a72#), the encoder focus on small patch of images and lose the big picture, which implies that the segmentation will be messy/blurry, not very precise. Skipping connections (performed thanks to encoder-decoder linking) allows to counter that effect by combining the two layers by addition.

![Putty](Skip.PNG)

### Why a FCN ?
The FCN is widely used for semantic segmentation (looking at the image and being able to distinguish/identify each object in the scene).

Using a Fully CONNECTED Network would not be good: as said in the lecture, it could "recognize a hot dog, but not a hot dog in a plate". Here, the hero would not be correctly recognized in the scene.

Using a classic convolution layer would not allow to recognize correctly the hero at different distances: the hero would be of different sizes and therefore not recognized (no skipping).

![Putty](FCN.PNG)

## Learning Parameters Understanding (first idea)

We explained above the FCN parameters.
How about the meaning of the Learning parameters ?

The learning rate is the size of the jump/steps:
- Smaller learning rate
  - :-) More precise/stable
  - :-( Much longer computation

The batch size (number of images processed per batch):
- Too big batch size
  - :-( GPU fail (out of memory?)

The number of epochs is the number of iterations:
- More epochs
  - :-) Accuracy Improvements
  - :-( (Linearly?) longer computation

The number of stepsper epoch is the number of changes per epoch (closely linked to the learning rate):
- More steps
  - :-) Accuracy Improvements
  - :-) (Linearly?) longer computation
Note: in the course, it is recommended to use the number of images divided by the batch size

The number of workers is linked to the number of cores in the CPU.

Therefore, based on the course and the above understanding:

```python
learning_rate = 0.001
batch_size = 32
num_epochs = 200
steps_per_epoch = 4131 / batch_size
validation_steps = 1184 / batch_size
workers = 2
```

## Testing local
Before throwing away money into Cloud infrastructure for a non working script, I first tested my code with a 0.01 learning_rate and 5 epochs only on my old laptop. After some debug (mostly pip install libraries) and a night of computation, I got low result but the assurance that the script was working.
I therefore tested on Amazon Cloud.

## Testing on cloud
After following the GPU Cloud instructions on the course, I connected with converted private key and Putty to:
```
ubuntu@ec2-34-245-36-199.eu-west-1.compute.amazonaws.com
Port 22
```
![Putty](putty1.PNG)
![Putty](putty2.PNG)

After a few trials, my best parameters were:
```python
learning_rate = 0.001
batch_size = 32
num_epochs = 200
steps_per_epoch = 4131 / batch_size
validation_steps = 1184 / batch_size
workers = 2
```

## Results discussion, future enhancements

I achieved a score of:
```python
# And the final grade score is
final_score = final_IoU * weight
print(final_score)
0.460961615459
```

To go more into results, let's look at different situations

---
![Putty](epoch200.PNG)
Looking at the training and validation curve, it seems that there is no use to go beyond 100 epochs.

---
![expensive](expensive.PNG)
I could have save some bucks on Amazon Cloud :-)

---
![expensive](follow1.PNG)
![expensive](follow2.PNG)
As the final score says, the results seems pretty good as the hero is correctly followed and distinguished from other people, which are correctly also correctly identified.

---
![expensive](patrol.PNG)
At patrol, other humans are detected but not misidentified as our hero. That's a good thing !

---

![expensive](patroltarget.PNG)
I was quite impressed when finding out that the UAV could recognize the hero from far away while at patrol.

## Future Enhancements
![expensive](somefail1.PNG)
Gladly enough (I would start to be afraid), there are failures. Here above, the UAV misidentifies partly a non-hero as hero.

Obviously, FCN are not reserved to humans detection only. By adapting the training data to other kind of objects/animals, we could also detect dogs, cats, birds, dinosaurs, ...

I could also improve my results by unmixing up my depth of enconder/decoder filters as explained in this report.

Fine tuning the parameters might improve the results but I think that more training data (especially with obstacle in the view, or at different distances...) would be of a greater interest.

Finally, to actually do a real-world application, we should add some stabilisation pre-processing, contrast improvements and such due to the nature of UAV (moving accross winds, etc).
