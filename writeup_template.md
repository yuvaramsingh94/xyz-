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

You're reading it! :)

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.
The color selection of the obstacles was quite easy it's basically the opposite of the navigable path.
```
def obstacles(img, rgb_thresh=(160, 160, 160)):
    color_select = np.zeros_like(img[:,:,0])
    below_thresh = (img[:,:,0] < rgb_thresh[0]) \
                & (img[:,:,1] < rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    color_select[below_thresh]=1
    return color_select
```
For rock samples, I had to analyse the color range for yellow I would need to indicate and threshold for. Using the matplot lib interactive graph was good because the RGB values were available.

```
def rock_samples(img):
    color_select = np.zeros_like(img[:,:,0])
    rock_thresh = (img[:,:,0] >= 140) & (img[:,:,0] <= 187) \
                & (img[:,:,1] >= 108) & (img[:,:,1] <= 145)\
                & (img[:,:,2] >= 0) & (img[:,:,2] <= 20)
    color_select[rock_thresh]=1
    return color_select
```

### Perspective Transform


![alt tag](https://github.com/fola95/Udacity-Rover-Project/blob/master/code/graphs/rock-transform.png "Perspective transform")

#### 2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 
The process image function was a combination of the previous steps done in the jupyter notebook.
The steps are:
1. Apply perspective tansform on the said image
2. With the transformed image perform thresholding for objects of interest: navigable path, obstacles and rocks
3. Once we have the threshold values we convert these to rover coordinates
4. Rover coordinates are then converted to world map coordinates for mapping
5. The RGB values for each object to ensure contrast on the world map. Obstacles - RED, Navigable path - Green

Please find video below:

https://github.com/fola95/Udacity-Rover-Project/blob/master/output/mapping.mp4

```

# Define a function to pass stored images to
# reading rover position and yaw angle from csv file
# This function will be used by moviepy to create an output video

def get_source_coords_ref():
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    return source

def get_destination_coord_ref(image):
    dst_size = 5 
    bottom_offset = 6
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])  
    return destination


def process_image(img):
    # Example of how to use the Databucket() object defined above
    # to print the current x, y and yaw values 
    # print(data.xpos[data.count], data.ypos[data.count], data.yaw[data.count])

    # TODO: 
    # 1) Define source and destination points for perspective transform
    # 2) Apply perspective transform
    src = get_source_coords_ref()
    destination = get_destination_coord_ref(img)
    
    
    warped, mask = perspect_transform(img, src, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    path_thresh = path(warped)
    obstacles_thresh= obstacles(warped)
    rock_thresh= rock_samples(warped)
    
    # 4) Convert thresholded image pixel values to rover-centric coords
    path_x, path_y = rover_coords(path_thresh)
    obs_x, obs_y = rover_coords(obstacles_thresh)
    rock_x, rock_y = rover_coords(rock_thresh)
    
  
    
    # 5) Convert rover-centric pixel values to world coords
    size= 200;
    
    #print(data.yaw)
    navigable_x_world, navigable_y_world= pix_to_world(path_x, path_y, data.xpos[data.count], data.ypos[data.count], data.yaw[data.count], data.worldmap.shape[0], 10)
    obstacle_x_world, obstacle_y_world= pix_to_world(obs_x, obs_y, data.xpos[data.count], data.ypos[data.count], data.yaw[data.count], data.worldmap.shape[0], 10)
    rock_x_world, rock_y_world= pix_to_world(rock_x, rock_y, data.xpos[data.count], data.ypos[data.count], data.yaw[data.count], data.worldmap.shape[0], 10)
    
    # 6) Update worldmap (to be displayed on right side of screen)
    
    data.worldmap[obstacle_y_world, obstacle_x_world, 0] =255
    data.worldmap[rock_y_world, rock_x_world, 1] = 255
    data.worldmap[navigable_y_world, navigable_x_world, 2] = 255
    
    nav_pix = data.worldmap[:, :, 2]>0
    data.worldmap[nav_pix,0]=0
    
    obstacle_pix = data.worldmap[:, :, 1]>0
    data.worldmap[obstacle_pix,0]=0

    # 7) Make a mosaic image, below is some example code
        # First create a blank image (can be whatever shape you like)
    output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1]*2, 3))
        # Next you can populate regions of the image with various output
        # Here I'm putting the original image in the upper left hand corner
    output_image[0:img.shape[0], 0:img.shape[1]] = img

        # Let's create more images to add to the mosaic, first a warped image
    warped, mask = perspect_transform(img, source, destination)
        # Add the warped image in the upper right hand corner
    output_image[0:img.shape[0], img.shape[1]:] = warped

        # Overlay worldmap with ground truth map
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.5, 0)
        # Flip map overlay so y-axis points upward and add to output_image 
    output_image[img.shape[0]:, 0:data.worldmap.shape[1]] = np.flipud(map_add)


        # Then putting some text over the image
    cv2.putText(output_image,"Analysis of video!", (20, 20), 
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    if data.count < len(data.images) - 1:
        data.count += 1 # Keep track of the index in the Databucket()
    
    return output_image


#out = process_image(grid_img)
#out.tobytes()


```

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

## perception.py
We combined th analysis done on the python notebook to create the perception steps.
Key things needed were a thresholded image of the path, obstacles, and rocks.
After which we pass this on to the rover to be aware of what each pixel means.
These pixels were then converted into world map coordinates and also given different colors.
We also calculate the polar coordinates of the navigable path to tell the rover what direction to move given the angles of the navigable terrain.

To improve fidelity I took note of Rover.pitch and Rover.roll that were they were greater than or equal to  0.5 the worldmap should not be updated. This would take care of cases where the rover was doing some heavy turns and its image would not be the best quality.

```
def path(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def obstacles(img, rgb_thresh=(160, 160, 160)):
    color_select = np.zeros_like(img[:,:,0])
    below_thresh = (img[:,:,0] < rgb_thresh[0]) \
                & (img[:,:,1] < rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    color_select[below_thresh]=1
    return color_select

def rock_samples(img):
    color_select = np.zeros_like(img[:,:,0])
    rock_thresh = (img[:,:,0] >= 140) & (img[:,:,0] <= 187) \
                & (img[:,:,1] >= 108) & (img[:,:,1] <= 145)\
                & (img[:,:,2] >= 0) & (img[:,:,2] <= 20)
    color_select[rock_thresh]=1
    return color_select
    

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    img = Rover.img

  
    src = get_source_coords_ref()
    dest = get_destination_coord_ref(img)
    
    # 2) Apply perspective transform
    warped, mask = perspect_transform(img, src, dest)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    
    rock_thresh = rock_samples(warped)
    obstacles_thresh = obstacles(warped)
    path_thresh = path(warped)
    
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,0] = obstacles_thresh*255
    Rover.vision_image[:,:,1] = rock_thresh*255
    Rover.vision_image[:,:,2] = path_thresh*255

    # 5) Convert map image pixel values to rover-centric coords
    path_x, path_y = rover_coords(path_thresh)
    obs_x, obs_y = rover_coords(obstacles_thresh)
    rock_x, rock_y = rover_coords(rock_thresh)
    
    # 6) Convert rover-centric pixel values to world coordinates
    navigable_x_world, navigable_y_world= pix_to_world(path_x, path_y, Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.worldmap.shape[0], 10)
    obstacle_x_world, obstacle_y_world= pix_to_world(obs_x, obs_y, Rover.pos[0],  Rover.pos[1], Rover.yaw, Rover.worldmap.shape[0], 10)
    rock_x_world, rock_y_world= pix_to_world(rock_x, rock_y, Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.worldmap.shape[0], 10)
    
    # 7) Update Rover worldmap (to be displayed on right side of screen)
    if Rover.pitch<0.5 and  Rover.roll<0.5:                
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] =255
        Rover.worldmap[rock_y_world, rock_x_world, 1] = 255
        Rover.worldmap[navigable_y_world, navigable_x_world, 2] = 255
    
    nav_pix = Rover.worldmap[:, :, 2]>0
    Rover.worldmap[nav_pix,0]=0
    
    rock_pix = Rover.worldmap[:, :, 1]>0
    Rover.worldmap[rock_pix,0]=0
    
    

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    
    dist, angles = to_polar_coords(path_x,path_y)
    Rover.nav_dists = dist
    Rover.nav_angles = angles
    
    
    return Rover
```

## decision.py
Here I modularized the code to get a better understanding of what each step was doing.
We have two modes.
Rover could be stopped or moving forward.
For forward, where there was sufficient terrain, we would continue to move and adjust the steering angle using the mean navigable path polar coordinate angle previously given in the perception step. When navigable terrain is below threshold, we would stop and perform a continous turn to the right until the vision was enough to "go forward". Once able we update the rover state to forward and continue. 
```

def move_forward(Rover):
    # Set throttle back to stored value
    Rover.throttle = Rover.throttle_set
    # Release the brake
    Rover.brake = 0
    # Set steer to mean angle
    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
    Rover.mode = 'forward' 

def brake(Rover):
    Rover.throttle = 0
    Rover.brake = Rover.brake_set
    Rover.steer = 0

def decision_step(Rover):

    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward': 
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                
                # Set steering to average angle clipped to the range +/- 15
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)

                    # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                brake(Rover)
                
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 
                        
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    move_forward(Rover)
                    Rover.mode='forward'
                   
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover

```   
  




#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

As mentioned, I was able to adjust the world map based on roll and pitch angles to improve fidelity.

```
   if Rover.pitch<0.5 and  Rover.roll<0.5:                
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] =255
        Rover.worldmap[rock_y_world, rock_x_world, 1] = 255
        Rover.worldmap[navigable_y_world, navigable_x_world, 2] = 255

```


Going forward I would have loved to understand how to manipulate the nav angles to make my rover biased to the left wall.
I tried some things like making the Rover.steer=-15 and only adjust to normal steer ( mean navigable angle) when navigable pixels below 250 but greater than 50. But the Rover behaviour was worse than my final solution :) . A really interesting problem and would definitely like to learn how other students accomplish that. :).

Configurations: OS- Mac; Screen Res- 840 x 524; Graphics Quality - Good; fps was same as default (I see None in drive_rover.py)

I was able to map 57.5% of the environment at 84.8 percent fidelity


![alt tag](https://github.com/fola95/Udacity-Rover-Project/blob/master/code/graphs/autonomousShot.png)


