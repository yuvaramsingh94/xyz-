
# Project: Perception Pick & Place

## Exercise 1, 2 and 3 Pipeline Implemented

### Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

#### Point-cloud data is collected from the robots RGB-D camera and first filtered to remove data outliers with the use of a nearest-neighbour, standard deviation filter.
![avatar](http://s12.sinaimg.cn/large/0034PgD7zy7jcI9yOvhdb&690)
#### After  Voxel Grid Downsampling, take in the cloud into a passthrough through Z axis(what's more, i can also use X or Y axis, and that's what  I did in the project).
![avatar](http://s10.sinaimg.cn/large/0034PgD7zy7jcHhsWxP69&690)
![avatar](http://s10.sinaimg.cn/large/0034PgD7zy7jcHhsWxP69&690)
![avatar](http://s5.sinaimg.cn/large/0034PgD7zy7jcHhvUiMa4&690)
####  RANSAC is an algorithm, that you can use to identify points in your dataset that belong to a particular model.Here the inliers is table and the outliers is all other objects. 
![avatar](http://s10.sinaimg.cn/large/0034PgD7zy7jcHhsWxP69&690)


#### Output of Voxel Grid data point clouds:
![avatar](http://s16.sinaimg.cn/large/0034PgD7zy7jcIjQD0P5f&690)


#### Output of Table point clouds:
![avatar](http://s16.sinaimg.cn/large/0034PgD7zy7jcIkvThZ7f&690)


#### Output of Objects point clouds:
![avatar](http://s7.sinaimg.cn/large/0034PgD7zy7jcIkjCho76&690)



### Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.

#### After the above code, Euclidean Clustering is needed to  locate and segment each cluster for classification. 
#### Below is an example of the clustered outliers point cloud：
![avatar](http://s15.sinaimg.cn/large/0034PgD7zy7jcIjw4PQce&690 )

### Complete Exercise 3 Steps. Features extracted and SVM trained. Object recognition implemented.

#### After the clusters have been located, what I need to do is to use SVM to classify the totally 6 objects whose features are already generated. 
#### I have to point out that although in the Exercise , other complex kernels seems to performed better than linear kernel. But there's a very deadly problem: Overfitting! I tried to use rbf kernel which seems good . However, it can always misread other objects as "book". The performance in the confusion matrices, the accuracy and score are also the  way to see how the model perform. So, generally speaking, I think that linear kernel is the best in SVM.svc() function. Thanks to a friend of mine reminded me of the conclusion of Deeplearning.ai : when features are in a large amount but samples are lack, linear kernel is often better. Because in such large features, it is probably that the data are linearly separable. Although I think that rbf can also express the result of linear kernel in a way, but , as I said above, rbf is easy to overfit the data.
#### So my SVM parameters are like this(seems so simply but it took me some times to realize that, I used rbf before and it also perform well in confusion matrices):

                                           C=1.5,kernel='linear'* others are default
#### See my previous rbf confusion matrices and score:
![avatar](http://s8.sinaimg.cn/large/0034PgD7zy7jcIjyK4Df7&690)
![avatar](http://s9.sinaimg.cn/large/0034PgD7zy7jcIkmD4A48&690 )


That seems good ! But in fact, just in the test_world 1, it misrecognized to "soap" as "book"!!!!!!

#### My linear confusion matrices and score:
![avatar](http://s5.sinaimg.cn/large/0034PgD7zy7jcIjCKfG84&690 )
![avatar](http://s4.sinaimg.cn/large/0034PgD7zy7jcIkpnLZ73&690)



## Pick and Place Setup

The test environment's cover three different combinations of various objects. The potential objects to be observed are:

1.    biscuits
2.    soap
3.    soap2
4.    book
5.   glue
6.    sticky_notes
7.    snacks
8.    eraser


#### Screenshot of object detection in world 1:   3/3(100%)
![avatar](http://s6.sinaimg.cn/large/0034PgD7zy7jcIkzGfPa5&690)


#### Screenshot of object detection in world 2:   5/5(100%)
![avatar](http://s11.sinaimg.cn/large/0034PgD7zy7jcIkMRQu4a&690)


#### Screenshot of object detection in world 3:   8/8(100%)
![avatar](http://s1.sinaimg.cn/large/0034PgD7zy7jcIkNgswe0&690)



```python

```
