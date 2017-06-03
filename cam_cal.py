import numpy as np
import cv2
import glob
import pickle   

# read and make a list of all calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

obj_points = [] #real points
img_points = [] #pixel location in image

n_x = 9 # number of inner corners along x axis 
n_y = 6 # number of inner corners along y axis 

objp = np.zeros((n_x * n_y,3),np.float32)
# real x, y coordiantes.  z is always zero. it is the same for all images.
objp[:,:2] = np.mgrid[0:n_x, 0:n_y].T.reshape(-1,2)

for fname in images:
    print('processing ' + fname)
    #read image
    img = cv2.imread(fname)
    #img = cv2.imread(fname)
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #find the the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (n_x,n_y), None)
    #if corners found add object points and image points
    if ret == True:
        img_points.append(corners)
        obj_points.append(objp)
        #draw and display corner point to double check
        img = cv2.drawChessboardCorners(img, (n_x,n_y), corners, ret)
        write_name = fname[:-4] + '_cal.jpg'
        cv2.imwrite(write_name, img)
    
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)   
cal_dict = {}
cal_dict['mtx'] = mtx
cal_dict['dist'] = dist
# write calibration data to a pickle file
pickle.dump(cal_dict, open('camera_cal.pkl', 'wb'), -1)
