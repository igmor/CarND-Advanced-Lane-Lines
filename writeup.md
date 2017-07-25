## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./test_images/distorted.jpg "Distorted"
[image2]: ./test_images/undistorted.jpg "Unistorted"
[image3]: ./test_images/before_wrap50.png "Input frame"
[image4]: ./test_images/wraped50.png "Wraped"
[image5]: ./test_images/gradient50.png "X, Y Gradient"
[image6]: ./test_images/mask50.png "Color treshold mask"
[image7]: ./test_images/data_hist50.png "Binary pictureHistogram"
[image8]: ./test_images/before_unwrap50.png "Polylines fit"
[image9]: ./test_images/unwraped50.png "Unwraped lane"
[image10]: ./test_images/result50.png "Output framee"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Camera matrix and distortion coefficients. 

The code for this step is contained in the first code cell of the IPython notebook located in "./P1.ipynb"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function:

![alt text][image1]

and obtained this result: 

![alt text][image2]

### Pipeline (single images)
#### 1. For every input frame:

![alt text][image3]

#### 2. Color and gradient tresholding

I used a combination of color and gradient thresholds to generate a binary image, see `def color_and_gradient_pipeline` function. 
First I blurred the image with gaussian blurring filter to remove the noise, then
I computed color treshold masks to select left yellow lane and right white line, the color mask is one of the constituents in a result binary image, all color tresholding is done in HSV color space that is resilient to bright or dark color saturations: 

![alt text][image6]

The image then got transformed into HLS color space as L-channel is the most stable for gradient tresholding, then Sobol gradient operators were computed for X and Y with correspoding tresholds and kernel size, the result of this operation is combined into mask:

`final_binary[((sybinary >= 0.5) | (sxbinary == 1) ) & (mask_lane >= 0.5)] = 1`:


![alt text][image5]
 

#### 3. Perspective transform 

The code for my perspective transform includes a function called `warp_image()`, which appears in the 2rd code cell of the IPython notebook).  The `warp_image()` function takes as inputs an image (`img`) and defines source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[200./1280*w,720./720*h],
                  [453./1280*w,547./720*h],
                  [835./1280*w,547./720*h],
                  [1100./1280*w,720./720*h]])
dst = np.float32([[(w-x)/2.,h],
                  [(w-x)/2.,0.82*h],
                  [(w+x)/2.,0.82*h],
                  [(w+x)/2.,h]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 320, 720      | 
| 453, 547      | 320, 590      |
| 835, 547      | 960, 590      |
| 1100, 720     | 960, 720      |


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image:

![alt text][image4]

#### 4. Lane detection and visualization

I built a histogram of intensity pixels on a wraped image of a gradient:

![alt text][image7]

and used a window search method around maximum peaks of intensities to find non zero pixels in every window, that gave me nwindows=9 points on the left and right lanes. Then I used `numpy.polyfit` function to fit a polynom of 2d order into the points for the left and right subset accordingly:

![alt text][image8]


#### 5. Lane curvature position of the vehicle with respect to center.

`find_lanes` function returns left and right fit polynoms of a 2d degree, then `calc_curvature` function computes curvature and car's offset from the center in a following manner:

```python
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/(rightx[0] - leftx[0]) # meters per pixel in x dimension
```

we define pixels to real world meters conversion coefficients and then fit new polynoms this time in a real world space, then compute curvatures for the left and right polynoms using standard derivatives forlmula. The average of a right and left curvatures will give us the result curvature on a lane. 

For a car's offset from the center we compute `lane_center_x = leftx[0] + (rightx[0] - leftx[0])/2` the detected lane center and how far is it from the image's center. 

#### 6. Output image

The result lane gets unwraped by using `unwrap_image` function where we used a reverse procedure of a `wrap_image` function:

![alt text][image9]

 and get blended into original frame:

![alt text][image10]


---

### Pipeline (video)

#### 1. 

Here's a project video [link to my video result](./test_videos_output/project_video.mp4)

#### 2. 

Here's a challenge video [link to my challenge video result](./test_videos_output/challenge_video.mp4)

---

### Discussion


#### 1.

The pipeline works reasonably ok in scenes with decent amount of shades, reasonable amount of lights, lanes going more or less straigt into a car's camera at 90 degrees angle. To make curvature computations more stable I tried to smooth out new lane findings using a well known smoothing technique: `lane = 0.9*lane + 0.1*new_lane`. That approach provides some robustness at a cost of the entire solution not being able to adapt to quick lane changes like harder challenge video.

#### 2.

I also had to implemented number of tricks to cover challenge video project, i.e:
I had to introduce color tresholding in HSV space for yellow and white lane. So obviosly the method will fail if left lane is not yellow.

#### 3.

There is part of the challenge video where car goes under the bridge into a very dark spot, that drives all the gradients and color treshold crazy. To make the algorithm more robust I track the area of a lane polygon and if it starts exceeding rapidly I stop accepting lane changes until things get back to normal again. That means during some period of time (when a car goes through these tough scenes) we can't reliably detect lanes, so the error could be a bit higher then usual.  