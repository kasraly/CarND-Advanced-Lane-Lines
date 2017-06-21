
# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/cal_distorted.jpg "Distorted image"
[image2]: ./output_images/cal_undistorted.jpg "Undistorted image"
[image3]: ./output_images/test_image.jpg "Test image (undistorted)"
[image4]: ./output_images/binary.jpg "Edge detected binary image"
[image5]: ./output_images/perspecrive_change.jpg "Perspective changed"
[image6]: ./output_images/detected_lines.jpg "Fitted polynomial lines"
[image7]: ./output_images/search_for_lines.jpg "Searching for lines"
[image8]: ./output_images/pipeline_image.jpg "Pipeline image"
[video1]: ./project_video_tracked.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view)

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation

---

### Writeup / README

### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one

You're reading it!

## Camera Calibration

### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I utilized the openCV camera calibration tools based on chessboard images. The code for this part is in the file `cam_cal.py`
The chessboard images were read from file and converted to grayscale. then using the 'cv2.findChessboardCorners'

I am keeping the coordinates of the chessboard corners in `obj_points` and `img_points`. `img_points` holds the (x, y) pixel coordinates of corners in the images. `obj_points` hold the real (x, y, z) coordinate of the chessboard corners. For the chessboard, we consider z=0 for all point and (x, y) coordinates will be a grid. I utilized the variable `objp` that holds a template for real coordinates of one chessboard corners. I use opencv `cv2.findChessboardCorners` function to find the chessboard corners in a test image. Every time all chessboard corners in a test image is successfully detect, a `objp` template will be appended to `obj_points` as a copy of chessboard real coordinates . Also `img_points` will be appended with the (x, y) pixel position of each of the corners obtained from `cv2.findChessboardCorners`. Additionally, I draw the detected corners on the image using `cv2.drawChessboardCorners()` to check the locations and validate them.

I then used the `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to one of the images using the `cv2.undistort()` function and obtained the result below. the left image is the original image, abd the right image is distortion corrected. This imagae was not included in the calibration, since not all the corners were visible and corner detection was failed.

The left image below is an original calibration image with the distortion. the right image show the distortion corrected version of the same image. 

![alt text][image1]![alt text][image2]

## Pipeline (single images)

### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image3]

### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result

I experimented with various gradient approaches and color thresholds to find an effective way to find the lanes marks. The binary image is calculated in lines #84 to #88 of the image process pipeline in `image_gen.py` file.
I used 2 approaches to find lane lines. 1) the function `grd_max_col_thresh` (lines #42 to #50 in `image_gen.py`) which takes the gradient along the X axis for the separate color channels and takes the maximum of the absolute value of three channels as the gradient of that pixel. 2) the function `col_thresh` (lines #52 to #62 in `image_gen.py`) which applies a threshold to 'V' channel of 'HSV' color space and 'S' channel of 'HLS' color space. Then through 'AND' operation, only the pixels that satisfy the threshold for both color space will be activated.

Then the output of these two functions will be combined through 'OR' operation. Here's an example of my output for this step.

![alt text][image4]

### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for calculating the martrices for perspective change appear in lines #13 through #30 of `image_gen.py` file. I defined four parameters to represent the trapezoid in the original camera image. `bot_width` defines the number of pixels of the bottom of trapezoid. `hood_heigth` defines the pixels at the bottom of image for the car hood. `top_bot_ratio` is the ratio between trapezoid top width and bottom width. `trap_height` is the height of the trapezoid. The shape in the bird's eye view perspective is a rectangle from top to bottom of the image with some padding from sides of the image.

I found the proper trapezoid through trial and error, and by examining the resulting destination image and ensure the lane lines are parallel. This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 155, 673      | 250, 720        |
| 1125, 673      | 1030, 720      |
| 589.56, 446     | 250, 0      |
| 590.44, 446      | 1030, 0        |

I verified that my perspective transform was working by examining the bird's eye view image. As can be seen from the image below, the horizontal distance between the lanes at the top and bottom of the image are very similar, showing that the two lanes are parallel.

![alt text][image5]

### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial

The file `line.py` handles finding the lane lines based on the binary image and fitting a polynomial. The main function is `find_lanes` at line #107. This function assumes we already know the left and right lane polynomials from previous frame. We first find all active pixels in the binary image that are within specific margin of the previously found lane polynomials (lines #114 to #121). I did revert the image Y axis to have the pixels with zero Y coordinates at the bottom of the image (see line #116 in `line.py`). 
Then I fitted a second order polynomial to each left and right set of pixels (lines #123 to #130 in `line.py`). Image below show the left and right lines found on the test image. The green are shows the search margin and red and blue pixels are active pixels for left and right lane lines. The yellow line shows the fitted polynomial.

![alt text][image6]

After that we calculate the left and right lines curvature at the bottom of the image, y-value = 0. I checked to ensure the found lines are valid. For this, I compared the difference between inverse of line and when it was above certain threshold, I would assume the lines are not valid. Additionally, if the left line starts on the right side of the picture or the right line is found in the left side of the picture the lines would be invalid (lines #139 to #140 in `line.py`).

I used a low pass filter for the lines polynomials to overcome sudden changes and noises in the image. When a new fitted polynomial is found, it would only change the last known lane lines by 20% (see line #146 to #148 in `line.py`).

#### Not having valid lines from previous frames

As mentioned above the `find_lanes` function assumes we have an estimate for lane position from previous frames. When we do not have a valid line from previous frame, we need to search for the lines in the whole image. The function `search_for_lines` (line #28 in `line.py`) will perform this task. I followed a similar approach described in project material. I obtained the peak of the lower half of the image histogram. then searched for the active pixels from that location moving up. image below show the search boxes for the test image.

![alt text][image7]

### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center

I employed the course notes and the tutorial [here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) to calculate the curvature of a second order polynomial. I calculated the curvature at the bottom of the image (see lines #178 to #188 in `image_gen.py`). Since the polynomials are in pixels, I converted the x and y axis from pixel to real-world meters by estimating the number of pixels corresponding to lateral and longitudinal distance from lane markings.

I calculated the car drift from centre of the lane in lines #219 and #220 of the `image_gen.py` file. Assuming the camera is installed in the centre of car and positioned to look exactly forward, the centre of the image corresponds to middle of the car. By averaging the location where left and right lines pass the bottom of the image, we can estimate the center of the lane in the image. By computing the difference between centre of image and pixel corresponding to centre of lane, we can estimate the car drift from centre of the lane. Of course this difference is in pixels and is converted to meters.

### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

plotting the lines on the original images is implemented as part of the image processing pipeline. I implemented this step in lines #131 through #228 in my code in `image_gen.py` in the function `process_image()`.  Here is an example of my result on a test image. The left line is marked in red and right line is marked in blue. A small version of the lines found on the binary image is also shown on the top right corner. The line curvatures and drift from centre of the lane is written on the top left of the image. positive values represent position toward right (curve toward right, and vehicle to the right of the lane centre) and negative values represent position toward left.

![alt text][image8]

---

## Pipeline (video)

### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a the video output of my algorithm.
![alt text][video1]

---

## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The first step for finding the lane is to create the binary image. The test images had fairly good lighting condition and finding the lines were not hard, however many condition cause issues with finding the lane line pixel. Examples include night time, excessive shadows, pavement color, etc. It is To have a more robust detection of line pixels, one possible approach would be to have adaptive processing from the previous frame. Effectively, updating the thresholds and coefficients based on the region that previous lines were found.

The next step in finding the lines is perspective change. This step is the most sensitive part. Slight error in estimating the road surface in the image, can significantly decrease the performance of lane finding. Our assumption is that the orientation of the camera with respect to road will not change; however, there are numerous reasons that this assumption does not hold. First, the car moves up and down with road bumps changing the camera orientation. Second, the road itself is not flat and lines in distance will be skewed in the bird's eye view perspective. 
