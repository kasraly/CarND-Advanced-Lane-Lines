import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, margin=100, minpix=50, nwindows=12, ym_per_pix=12/200, xm_per_pix=3.7/540):
        # was the line detected in the last iteration?
        self.lanes_valid = False  
        #polynomial coefficients for the most recent fit
        self.left_fit = None
        self.right_fit = None
        #polynomial coefficients for the previous fit
        self.last_left_fit = None
        self.last_right_fit = None
        # margin of search for the lane pixels
        self.margin = margin
        # minimum number of pixels found to recenter window
        self.minpix = minpix
        # Choose the number of sliding windows
        self.nwindows = nwindows

        self.ym_per_pix = ym_per_pix # meters per pixel in y dimension
        self.xm_per_pix = xm_per_pix # meters per pixel in x dimension



    def search_for_lines(self, img):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((img, img, img))

        # Choose the number of sliding windows
        nwindows = self.nwindows
        # Set height of windows
        window_height = np.int(img.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = (img.shape[0]-1) - np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = self.margin
        # Set minimum number of pixels found to recenter window
        minpix = self.minpix
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        d_leftx_current = 0
        d_rightx_current = 0
        
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = window*window_height
            win_y_high = (window+1)*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,(img.shape[0]-1) - win_y_low),(win_xleft_high,(img.shape[0]-1) - win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,(img.shape[0]-1) - win_y_low),(win_xright_high,(img.shape[0]-1) - win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                d_leftx_current = np.int(np.mean(nonzerox[good_left_inds])) - leftx_current
            if len(good_right_inds) > minpix:        
                d_rightx_current = np.int(np.mean(nonzerox[good_right_inds])) - rightx_current
            rightx_current = rightx_current + d_rightx_current
            leftx_current = leftx_current + d_leftx_current

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        self.left_fit = left_fit
        self.right_fit = right_fit

        return left_lane_inds, right_lane_inds, out_img


    def find_lines(self, img):
        if self.lanes_valid==False:
            self.search_for_lines(img)

        left_fit = self.left_fit
        right_fit = self.right_fit

        margin = self.margin
        nonzero = img.nonzero()
        nonzeroy = (img.shape[0]-1) - np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) 
                          & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) 
                           & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit_new = np.polyfit(lefty, leftx, 2)
        right_fit_new = np.polyfit(righty, rightx, 2)
        
        left_fit_cr = self.xm_per_pix*np.divide(left_fit_new,np.array([self.ym_per_pix**2,self.ym_per_pix,1]))
        right_fit_cr = self.xm_per_pix*np.divide(right_fit_new,np.array([self.ym_per_pix**2,self.ym_per_pix,1]))
        # Calculate the new radii of curvature
        y_eval = 0
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0]) * np.sign(left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0]) * np.sign(right_fit_cr[0])

        if abs(500/left_curverad - 500/right_curverad) > 1:
            self.lanes_valid = False
        elif left_fit[2] > img.shape[1]/2 or right_fit[2] < img.shape[1]/2:
            self.lanes_valid = False
        else:
            self.lanes_valid = True

        alpha = 0.2
        left_fit = left_fit + alpha * (left_fit_new-left_fit)
        right_fit = right_fit + alpha * (right_fit_new-right_fit)
        self.left_fit = left_fit
        self.right_fit = right_fit
        
        return left_lane_inds, right_lane_inds