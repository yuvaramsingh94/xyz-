## Project: 3D Perception

**The goals / steps to complete this project were the following:**  

1. Get familiar with the concept of Point Cloud Library (PCL).  
2. Implement data filtering techniques for image recognition from a RGB camera. 
3. Understand Random Sample Consensus (RANSAC) plane fitting for identifying the points of an identified model.
4. Perform clustering segmentation in the generated Point Cloud Library.
5. Extract object's features extracted and train a Support Vector Machine (SVM). 
6. Recognixe objects detected by the RGB camera.

[//]: # (Image References)
[image3]: ./image1.png
[image4]: ./image2.png

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points

### Summary

The proposed project was developed by:
1. The implementation of data filtering and RANSAC plane fitting.
2. The clustering or segmentation of detected objects.
3. The extraction of features from the object and training of SVM. 
4. The object's recognition via project implementation.  

Keep in mind that most of the techniques herein described were applied to data coming out of a RGB camera. Such data had to be trasmitted via ROS messages over specified topics and converted to PCL (and XYZ positions), and then back into ROS messages for its transmission in RViz. In general, the creation of ROS publishers and further instructions is implicit all over the project. 

#### Data filtering / RANSAC plane fitting
This section was part of Exercise 1. The `pcl_callback()` function had to include instructions to properly filter out the data coming from the RGB camera. Threen different techniques were used in this part before the PCL could be clustered:
1. Voxel Grid Downsampling to reduce the incoming amount of data. 
2. Passthrough filter to separate the table from the detectable objects. 
3. Noise filtering to eliminate outlier points (e.g., not belonging to any object)

On the other hand, the RANSAC technique was also used and it allowed to narrow down the recorded points by determining whether they belonged to a specific model or shape. This way, it was possible to separate the objects from the table itself by defining a plane as segmentation shape (line 91 of `project_template.py`) and whether any points were within a threshold distance. 

#### Clustering
This section was part of Exercise 2. After the filtering, it was necessary to create a point cloud for any different object present on the table. The way to do so was by defining clusters of data (or points) over the XYZ space, for which different methods exists in the literature (K-means clustering, DBSCAN algorithm, etc). 

In our case, a 'k-d tree' was used as a data structure to organize a number of points in a k-dimensions space. Such struture allows to determine the set of point neighbors to a specific location (or radius). The Eucliden-clustering technique follows such this principle and it was used in `project_template.py` to reduce the amount of processing time of the streamed point cloud (lines 105 to 117). A color was assigned to each point cloud afterwards to differentiate the detected objects (line 121 to 134). 

#### Object Features's Extraction
This section was part of Exercise 3. Once all of the objects were extracted and clustered, instructions to create recognizable objects from the generated point clouds were included in the `pcl_callback()` function (see `project_template.py`, lines 147 to 180).  The funtions `compute_color_histograms()` and `get_nomarls()` stand out as a way to get the feature vector necessary to get the Support Vector Machine algorithm implemented (lines 166 to 168 in `project_template.py`). 

In a nutshell, the algorithm allows to characterize the parameter space (PCL) of the detected objects into discrete classes by usign a 'trained' feature vector and label (e.g., the SVM is trained to recognize whether an image contains some specific "based on an input feature vector composed of color histograms" (slide 12, Lesson 19:Object Recognition)). That been said, once the training is applied to the whole parameter space, it is possible to characterize new objects for which only features exists and afterwards assign it to a specific class. After this point, if the SVM predictions are good, such training can be used for the object recognition phase.

The latest paragraph accounts for a series of functions distributed among the files `features.py`, `capture_features.py` and `train_svm.py`. Take into account however that the mentioned SVM training set maybe not be accurate if the characteristics from which such feature vector was extracted are not enough. In order to improve such model, more and better features could to be extracted for instance. Image below provides an example of the resulting confusion matrix for a number of features N=10 under the `test1.world` configuration. The same parameters were kept for both `test2.world` and `test3.world`. 

![image3]

### Project Implementation
The next impressions can be gathered from the project's implementation:
1. There's a high risk of overfitting the model if number of features N > 50. This comes out after several attempts were made for N (10, 30, 50, 100, 250). As the number increased, the reported precision while runnning `train_svm.py` increased as well (even 96% reported), although no improvement was observed in the number of objects detected (as reported in RViz). Information in Slack helped in determinining under which conditions this situation would appear.  

2. Changing the number of bins from 32 to 16 improved the overfitting situation as more items could be detected in RViz. However, a number below such limit reduced considerably the reported precision of the model while running `train_svm.py`.

3. Setting `using_hsv=True` in `compute_color_histograms()` improved both the reported precision of the trained model and the number of detected objects in RViz. As a matter of fact, the setting had to be included in both `features.py` and `project_template.py` in order to get consistent results while running `pick_place_project.launch`. 

4. Nevertheless, none of the results observed in Rviz (for any of the test worlds) provided a so called 'passing submission'. As a matter of fact, during `test1.world` case, several objects were detected at the begining (over 20) and decayed consequently to either of the 3 options (jumping from 'biscuits' to 'soap2' to 'soap') as shown in the image. As the number of objects increased per test case, so increased the number of objects detected, but it also decayed as simulation time went through. This behavior could be due to hardware limitations, but it is unknown until which point. 

![image4]

5. Can't personally say whether the data of either `output_1.yaml`, `output_2.yaml` or `output_3.yaml` is within acceptable limits. Any feedback regarding this issue would be of much help for future assignments. 

### Final Thoughts

I consider this project not to be easy, even though much of the code and its implementation was grabbed from the provided material in the lessons. On the other hand, information provided via the Slack channel helped in determining the errors that led to unexpected reported labels. 

I also consider the lectures lacksome directions regarding the tools I could've used to better grip the assignment and debug any issue. However, I consider this as a lack of experience from my side and just another part of the learning process.  


