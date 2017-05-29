import numpy as np
import cv2
import matplotlib as plt
import glob


# read and make a list of all calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

obj_points = [] #real points
img_points = [] #pixel location in image

n_x = 8 #chess board x size
n_y = 6 #chess board y size
objp = np.zeros((n_x*n_y,3),np.float32)
# real x, y coordiantes. z is always zeroit is the same for all images. 
objp[:,:2] = np.mgrid[0:n_x, 0:n_y].T.reshape[-1,2] 

for fname in images:
    
    #read image
    img = cv2.imread(fname)   
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #find the the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (n_x,n_y), None)

    #if corners found add object points and image points
    if ret == True:
        img_points.append(corners)
        obj_points.append(objp)

        #draw and display corner point to double check 
        img = cv2.drawChessboardCorners(img, (8,6), corners, ret)
        plt.imshow(img)
    
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

dst = cv2.undistort(img, mtx, dist, None, mtx)
