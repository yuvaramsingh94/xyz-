# Project Follow me

## Steps to complete the project:
Clone the project repo [here](https://github.com/udacity/RoboND-DeepLearning-Project)
Fill out the TODO's in the project code in the code/model_training.ipynb file.
Optimize your network and hyper-parameters.
Train your network and achieve an accuracy of 40% (0.4) using the Intersection over Union IoU metric.
Make a brief writeup report summarizing why you made the choices you did in building the network.
## Project Specification
Rubric points for this project are explained [here](https://review.udacity.com/#!/rubrics/1155/view).

## writeup
### Model architecture
In this project I used the following model. This is called the encoder decoder model and it is known to be effective in the area of ​​image segmentation. The most famous architecture is [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/), but this time I got the result I wanted without using a complicated architecture.
 1. InputLayer:**IN** 160x160x3, **OUT** 160x160x3
 2. SeparableConv2D:**IN** 160x160x3, **OUT** 80x80x32
 3. BatchNorm:**IN** 80x80x3, **OUT** 80x80x3
 4. SeparableConv2D:**IN** 80x80x32, **OUT** 40x40x64
 5. BatchNorm:**IN** 40x40x64, **OUT** 40x40x64
 6. SeparableConv2D:**IN** 40x40x64, **OUT** 20x20x128
 7. BatchNorm:**IN** 20x20x128, **OUT** 20x20x128
 8. Conv2D:**IN** 20x20x128, **OUT** 20x20x256
 9. BatchNorm:**IN** 20x20x256, **OUT** 20x20x258
 10. BilinearUpSampling2D:**IN** 20x20:256, **OUT** 40x40x256
 11. Concatnate:**IN** 10 and 5, **OUT** 40x40x320
 12. SeparableConv2D:**IN** 40x40x320, **OUT** 40x40x128
 13. BatchNorm:**IN** 40x40x128, **OUT** 40x40x128
 14. BilinearUpSampling2D:**IN** 40x40x128, **OUT** 80x80x128
 15. Concatnate:**IN** 14 and 3, **OUT** 80x80x160
 16. SeparableConv2D:**IN** 80x80x160, **OUT** 80x80x64
 17. BatchNorm:**IN** 80x80x64, **OUT** 80x80x64
 18. BilinearUpSampling2D:**IN** 80x80x64, **OUT** 160x160x64
 19. Concatnate:**IN** 18 and 1, **OUT** 160x160x67
 20. SeparableConv2D:**IN** 160x160x67, **OUT** 160x160x32
 21. BatchNorm:**IN** 160x160x32, **OUT** 160x160x32
 22. Conv2D:**IN** 160x160x32, **OUT** 160x160x3

Just to be afraid, a convolution network called CNN is used for each layer. This is a configuration often used in the area of ​​image processing. In addition, the batch normalization layer normalizes the variation of data in the batch, and learning can be advanced quickly. This is said that because the bias of data input for each layer decreases, the input data will be within a certain range, so convergence of learning becomes faster accordingly. Separable (Depthwise) The convolution layer is usually to calculate both the channel and the spatial position by brute force, calculate the output which was separately calculated, and shorten the calculation time by combining later It is aimed at. It is said to be a contrivance in calculation, it is said that the calculation result does not change with the convolution layer.

### Training
In the training phase, the most important thing is adjustment of hyper parameters. Originally if you use [hyperopt](https://github.com/hyperopt/hyperopt), [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), etc. for parameter adjustment, you can decide the value exhaustively and objectively, but this time it was not done because of calculation time. First, learning_rate, but tried 0.005, 0.0015, 0.001. Since it diverges when increasing the value, there is an aim to surely converge with a small value. I finally learned well with 0.005, so I used it. Next is epoch number, which is determined by how much learning converges. Since we converged at about 20 this time, we used that value. If it does not converge it will only stretch. For other parameters, we decided based on private GPU.

### Result
The dataset of this project is downloaded from [Here](https://classroom.udacity.com/nanodegrees/nd209/parts/09664d24-bdec-4e64-897a-d0f55e177f09/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/06dde5a5-a7a2-4636-940d-e844b36ddd27)
 * following_images : Images recognizing targets to follow.
 * patrol_with_targ : Images that we recognize including targets that follow and others.
 * patrol_non_targ : Image with no target.

 * following_images  
 ![image1](./docs/misc/following_images.png)

 * patrol_with_targ  
 ![image1](./docs/misc/patrol_with_targ.png)

 * patrol_non_targ  
 ![image1](./docs/misc/patrol_non_targ.png)

Finally I can got 0.41 score in my trial.

### Future improvement
 * The resolution of the image should be more high, e.g. the size of image, and the quality of image.
 * It is better to increase the dataset if you want to get more robustness in the other situation.
 * To improve the model, we need to use automatic hyper parameter optimization because the tuning is not quantitative.
