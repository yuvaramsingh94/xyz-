### How to Run the executable 

``` ./output <path to the video file > ```

eg 

``` ./output test_vid.mp4 ```

### About this code :

The main aim of this code is to determine whether the ego vehicle is moving or not based on the dashcam video . i approached this problem by finding a method to calculate the amount of change in pixel intensity which directly relates to the movement of either objects in the video or movement on the cam location .  there are lots of movable objects (other vehicles on road) which can change the background image event though the EGo vehicle is stationary . this code over comes the problem by having a threshold value which serves as a limit to determine whether the change recoderd by counting the disparity if false alaram or not . 

i did my coding firstly on python and then Re-coded it into a C++  code . i have good working experience on Python so i choose that language to produce a acceptable result . then i recoded everything in c++

### Code explanation 


firstly,The code takes three consecuting frames and analyse them . it perfimance ```absdiff``` which calculates the difference between frames (per element difference). this will let us know how much the image has changes . then we apply ```bitwise_and ```since  ```absdiff```function takes only two image at a time to process . 
```
Mat process(Mat t0,Mat t1, Mat t2) 
{
  Mat d1 , d2 , out;
  absdiff(t1,t2,d1);
  absdiff(t0,t1,d2);
  bitwise_and(d1, d2,out);
  return out;
}


```

we call this function from ``` countNonZero ``` function which basically returns the no of pixels whose values are greater than zero . this value directly correlate to the amount of change in the image . 

``` 
int totalDiff = countNonZero(process(t_minus, t, t_plus)); 
```

to make a video uncommand these lines 
```
 //VideoWriter video("out.avi",CV_FOURCC('M','J','P','G'),20, Size(640,480),true); // to write a video . 
 //video.write(frame);
```
