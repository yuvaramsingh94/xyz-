
## Project: Search and Sample Return

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

* Earlier parts of work was done in Lab jupyter notbook page online, so it was downloaded and used as Test one.
* Provided functions works quite well, but personally I prefer to write generalized affine transformations instead of specialized code. 
* HSV thresholding was added in order effectively find samples (`hsv_threshold(img, hsv_lower = (0, 0, 0), hsv_upper = (255, 255, 255))`). Initial thresholds was found by building simple one-channels histograms in both RGB and HSV spaces of provided images and trying to separate parts of multi-modal distribution. Further improvement was achieved by rendering images with mask and slowly tweaking parameters. After evaluating color thresholding on video, whole process was repeated on frames with clearly incorrect segmentation.

#### 2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 

* In order to render video, `process_image()` function was filled with image processing steps:
1) Warping image, in order to get rig of perspective projection
2) Color thresholding with different thresholds in order to find pixels, corresponding to obstacles, samples and navigable terrain.
3) Converting to rover coordinates.
4) Converting to world coordinates and scaling.
5) Filling worldmap with acquired data.
6) Generating supporting images

* Video was created on sample data online. And than was recreated locally in order to make sure that everything works.

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

Firstly, `perception_step()` code was mostly taken from `process_image()` and was adapted to existing framework. After evaluating resulting function in interactive environment, few flaws was found and addressed.
 1) Simple HSV thresholding was unable to recognize navigable terrain texture in high luminosity areas. RGB thresholding was failing in shaded areas, so I had to combine both approaches.
 2) Following tips from provided materials, pitch and roll thresholding was introduces in order not to update map with low quality data, as high pitch and roll would invalidate reverse perspective transform.
 3) Even with named thresholding there was still to many navigable terrain false positives, so pixels with high-confidence obstacles was forced to loose respective navigable terrain value.
 4) Point 3 was generally too harsh, so slight relaxation was added - pixels, proving themselves to be navigable, was allowed to slowly loose their obstacle rating.
 In this setup it was possible to achieve 95% mapping with 95% fidelity on manual controls.
 Also, rover-centered polar coordinates of navigable pixels was generated in order to make control decision-making possible. Named pixels was distance and angle thresholded as this method was found helpful while tuning controls in autonomous mode.
 
 Decision-making was addressed next. Provided code was able to achieve required fidelity, but failed to achieve required mapping, as was prone to find loop in navigable terrain and follow it. So, new "follow" mode was introduced. The main idea is that rover should somehow follow obstacles, effectively implementing "always turn left" maze crawling algorithm. It was done by binding steering to a navigable pixel percentage in defined sector. Accounting all thresholds, it's just truing to keep 15% of visible navigable terrain in sector between -0.15 and -0.5 radian (thresholded by distance between 20 and 150). This mode turned on by "forward" mode when named percentage in between 10% and 20% and aborted if it leaves 5%-25% range. Also it's able to make sudden turn if suck. Also "forward" mode was augmented to try slowly turn if stuck.

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  
 I was able to run described setup and record a run with external screen capture program. It was able to achieve declared (40% mapping, 60% fidelity) metric just under 78 seconds. Full run was aborted on second 389 with 96.1% mapped and 73.9% fidelity as rover stuck. All 6 rocks was located. None of the rocks was picked up.  Rover was unable to return to the starting point. Video of that run included in submission.
 With a little bit better parameter tuning rower would be able to not stuck and make full maze crawl, presumably improving mapping. Special "approach sample" state would make possible to collect located samples and more or less complete NASA challenge. Additional "return home" state would ensure success. Time is also subject of improvement here. Carefully tuned max speed, "follow" turn ratio and other thresholds could greatly improve mapping time. Also ability to skip confidently mapped parts on high speed would help significantly, as maze crawling involves lots of backtracking.  Using actual ML classifiers to determine navigable terrain and obstacles would also help a lot.

