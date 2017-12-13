## Project: Search and Sample Return
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook).
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands.
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./calibration_images/example_grid1.jpg
[image3]: ./calibration_images/example_rock1.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.
perspect_transform reads an image, and a defined calibration square and a defined square and returns the image transformed based on the defined square.  The idea is to transform this image as if we were looking at it from above so that it can be used as a map.  This function was modified to return the transformed image as well as an image mask to reference what pixels the robot can possibly see for use in the obstacle  map.

color_thresh reads an image and a threshold and returns a binary image of the image that meet the threshold for navigable terrain.  I modified this to return also binary images meeting the gold threshold , which I found to be >100 Red, > 100 Green, < 20 Blue.  Before watching the solution video, I was returning also the inverse of the navigable terrain, but it didn't produce a nice looking image.

#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result.
I defined the shape of the source and destination of the transform, and passed the parameter image to the perspect transform function.  

This returned the expected transformed image, as well as a binary image mask of the visible image.  

I then ran the transformed image through color_thresh, and got two binary images into the threshed array.  One for navigable terrain and the other for gold!  Per the suggestion of the solution video, I used the mask to find the inverse of the navigable terrain image as the obstacle map.  

These three images (navigable terrain, obstacle terrain, gold) were then processed to get arrays of x and y coordinates from the rover perspective.  The three resulting pairs of x,y coordinates were passed to the pix_to_world function along with the rover position, and yaw from the sample data to get the map translation of those coordinates.

Pixel values were added to the world map at these positions.  Obstacles went to the red channel, gold went to the green channel, and navigable terrain went to the blue channel.

I removed the transformed image, and added the robot vision to  the output image because I needed that to work for the simulator.


### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.
There wasn't much of a difference between the process_image() function above and the perception_step() function.

Differences:
Implemented the to_polar_coords function using the pixels valid for navigation that were previously identified.

The image was not considered valid for mapping unless the rover was relatively flat, that is with pitch and roll close to 0 angle.

I also added a "found gold" state to the rover, if any pixels in the image were found to meet the gold threshold, those pixels were used to calculate the nav_angles and dists.  I also set the state to "gold" so the rover can slow down and pick it up.

My rover was also getting into loops with a lot of open area, so I defined the pixel values of the world map to help the Rover distinguish between old and new terrain.

decision.py was modified to add the 'gold' mode.  The idea was if the rover sees gold, slow down, and navigate towards it.  If it gets close enough, stop and let it get picked up.

on line 42, I modified the steering of the rover to only consider pixels closer than 100 meters, and pixels that had a weight of < 1.25 the average of the weight.  It seemed like further away pixels got considered too much, and steered into rocks.  Filtering out pixel weights that have been visited many more times than the normal pixel in the video breaks us out of robot circles after a few laps.  It also makes the robot wobble a little if on a straightaway to look for rocks.

on line 70, I wanted the rover to see more navigable terrain in front of it before it started going again, so I clipped nav angles to show only between +- 5 degrees.

on line 91, if the rover picked up a rock, it would begin again from stop mode to find more navigable terrain before moving forward since rocks line the walls.

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

I get about 70% fidelity, and depending on the setup of the map 30-95% mapped.  My rover sometimes gets stuck on wall, or on the rocks in the middle, but it picks up gold!  I get 12FPS on 800x480 and Fastest graphics . . . time for an upgrade!

The most important part if I were to continue this project would be to figure out how to drive using input from both the camera, and the mapped image.  Sometimes the rocks really sneak up on you, and it doesn't seem like the transformed image even sees it.  I would also try to implement a "stuck" mode, where the rover would wiggle around or back up if there was throttle with no movement.  The final change, would be to modify the obstacle map so that not much would be mapped beyond the non-navigable terrain.  I think this was the cause of a lot of purple on the drivable terrain.
