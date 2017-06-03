import numpy as np
import cv2
import glob
import pickle   
import line
from moviepy.editor import VideoFileClip

# loading the camera calibration paramters
cal_dict = pickle.load(open('camera_cal.pkl', 'rb'))
mtx = cal_dict['mtx']
dist = cal_dict['dist']

# calculating image perspective change matrices
img_size = (1280, 720)
bot_width = 970 # bottom trapezoid width pixels 
hood_heigth = 47 # percent from top to bottom to avoid car hood
top_bot_ratio = 0.104 # ratio of top width to bottom width of trapezoid
trap_height = 227 # percent for trapezoid height
src = np.float32([[0.5*(img_size[0]-bot_width), img_size[1]-hood_heigth],
                    [0.5*(img_size[0]+bot_width), img_size[1]-hood_heigth],
                    [0.5*(img_size[0]-bot_width*top_bot_ratio), img_size[1]-hood_heigth-trap_height],
                    [0.5*(img_size[0]+bot_width*top_bot_ratio), img_size[1]-hood_heigth-trap_height]])
side_offset = 250
top_offset = 0
dst = np.float32([[side_offset, img_size[1]],
                    [img_size[0]-side_offset,img_size[1]],
                    [side_offset, top_offset],
                    [img_size[0]-side_offset,top_offset]])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

def grd_mag_thresh(img, ksize=3, thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, None, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, None, ksize=ksize)
    grd_mag= np.sqrt(sobel_x**2 + sobely**2)    
    grd_scaled = np.uint8(255*grd_mag/np.max(grd_mag))
    binary = np.zeros_like(grd_scaled)
    binary[(grd_scaled >= thresh[0]) & (grd_scaled <= thresh[1])] = 1
    return binary   

def grd_max_col_thresh(img, dir='x', thresh=(0,255)):
    if dir=='x':
        grd = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    else:
        grd = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    grd_max= np.max(np.abs(grd), axis=2)
    grd_scaled = np.uint8(255*grd_max/np.max(grd_max))
    binary = (grd_scaled >= thresh[0]) & (grd_scaled <= thresh[1])
    return binary   

def col_thresh(img, sthresh=(0,255), vthresh=(0,255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  
    V = hsv[:,:,2]
    binary_v = (V >= vthresh[0]) & (V <= vthresh[1])

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)  
    S = hls[:,:,2]
    binary_s = (S >= sthresh[0]) & (S <= sthresh[1])
    
    binary = binary_s & binary_v
    return binary   

def grd_dir_thresh(img, xy_ratio=1, ksize=3, thresh=(0,255)):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, None, ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, None, ksize=ksize)
    grd_max_comb = np.max(np.abs(sobelx), axis=2) + xy_ratio*np.max(np.abs(sobely), axis=2)
    grd_scaled = np.uint8(255*grd_max_comb/np.max(grd_max_comb))
    binary = (grd_scaled >= thresh[0]) & (grd_scaled <= thresh[1])
    return binary   

def grd_dir_thresh(img, xy_ratio=1, ksize=3, thresh=(0,255)):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, None, ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, None, ksize=ksize)
    grd_max_comb = np.max(np.abs(sobelx), axis=2) + xy_ratio*np.max(np.abs(sobely), axis=2)
    grd_scaled = np.uint8(255*grd_max_comb/np.max(grd_max_comb))
    binary = (grd_scaled >= thresh[0]) & (grd_scaled <= thresh[1])
    return binary   


def process_image(img):    
    img = cv2.undistort(img, mtx, dist, None, mtx)

    binary1 = grd_max_col_thresh(img, dir='x', thresh=(20, 100))
    #binary2 = grd_dir_thresh(img, 0, ksize=7, thresh=(20, 100))
    binary3 = col_thresh(img, (150,255), (120, 255))
    edge_detected = np.zeros_like(img[:,:,1])
    edge_detected[binary1 | binary3]=255

    #return np.dstack((edge_detected,edge_detected,edge_detected))

    warped = cv2.warpPerspective(edge_detected, M, img_size, flags=cv2.INTER_LINEAR)

    ## finding the line for the first time (or when lost confidence in previous lines)
    ## this will be called internally in the line class, no need to call it when actually
    ## processing the images to find lines
    #left_lane_inds, right_lane_inds, out_img = lane_tracker.search_for_lanes(warped)
    #left_fit = lane_tracker.left_fit
    #right_fit = lane_tracker.right_fit

    ## plotting the initial lines found
    #nonzero = warped.nonzero()
    #nonzeroy = np.array(nonzero[0])
    #nonzerox = np.array(nonzero[1])
    #curve_y = np.linspace(0, img.shape[0]-1, img.shape[0])
    #ploty = (img.shape[0]-1) - curve_y
    #left_fitx = left_fit[0]*curve_y**2 + left_fit[1]*curve_y + left_fit[2]
    #right_fitx = right_fit[0]*curve_y**2 + right_fit[1]*curve_y + right_fit[2]
    #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #lines_img = np.zeros_like(out_img)
    #line_width = 5
    #left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-line_width, ploty]))])
    #left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+line_width, ploty])))])
    #left_line_pts = np.hstack((left_line_window1, left_line_window2))
    #right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-line_width, ploty]))])
    #right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+line_width, ploty])))])
    #right_line_pts = np.hstack((right_line_window1, right_line_window2))
    #cv2.fillPoly(lines_img, np.int_([right_line_pts]), (0, 0, 255))
    #out_img = cv2.addWeighted(out_img, 1, lines_img, -1, 0)
    #cv2.fillPoly(lines_img, np.int_([left_line_pts]), (255, 255, 0))
    #cv2.fillPoly(lines_img, np.int_([right_line_pts]), (255, 255, 0))
    #out_img = cv2.addWeighted(out_img, 1, lines_img, 1, 0)
    
    # plotting the the lines found after searching the boundary of last line
    left_lane_inds, right_lane_inds = lane_tracker.find_lanes(warped)
    left_fit = lane_tracker.left_fit
    right_fit = lane_tracker.right_fit

    # Generate x and y values for plotting
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    curve_y = np.linspace(0, img.shape[0]-1, img.shape[0] )
    ploty = (img.shape[0]-1) - curve_y
    left_fitx = left_fit[0]*curve_y**2 + left_fit[1]*curve_y + left_fit[2]
    right_fitx = right_fit[0]*curve_y**2 + right_fit[1]*curve_y + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((warped, warped, warped))
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    margin = lane_tracker.margin
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    lines_img = np.zeros_like(out_img)
    line_width = 3
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-line_width, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+line_width, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-line_width, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+line_width, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    cv2.fillPoly(lines_img, np.int_([right_line_pts]), (0, 0, 255))
    out_img = cv2.addWeighted(out_img, 1, lines_img, -1, 0)
    cv2.fillPoly(lines_img, np.int_([left_line_pts]), (255, 255, 0))
    cv2.fillPoly(lines_img, np.int_([right_line_pts]), (255, 255, 0))
    out_img = cv2.addWeighted(out_img, 1, lines_img, 1, 0)

    detected_lines = out_img

    # drawing the lines in the original road image          
    y_eval = curve_y[img.shape[0]//2]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 20/500 # meters per pixel in y dimension
    xm_per_pix = 3.7/600 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = xm_per_pix*np.divide(left_fit,np.array([ym_per_pix**2,ym_per_pix,1]))
    right_fit_cr = xm_per_pix*np.divide(right_fit,np.array([ym_per_pix**2,ym_per_pix,1]))
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0]) * np.sign(left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0]) * np.sign(right_fit_cr[0])

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    line_width = 10
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-line_width, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+line_width, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-line_width, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+line_width, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    # I am subtracting these images, so used a inverted color
    cv2.fillPoly(color_warp, np.int_([pts]), (150, 0, 150))
    cv2.fillPoly(color_warp, np.int_([left_line_pts]), (0, 255, 255))
    cv2.fillPoly(color_warp, np.int_([right_line_pts]), (255, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    color_img = cv2.addWeighted(img, 1, newwarp, -0.3, 0)
    
    color_img[40:360,920:1240,:] = cv2.resize(detected_lines,(320,320))
    
    camera_x_offset = 0.1
    lane_centre_drift = xm_per_pix * ((left_fitx[0] + right_fitx[0])/2 - img.shape[1]/2) - camera_x_offset

    if lane_tracker.lanes_valid == True:
        text_color = (0, 255, 0)
    else:
        text_color = (255, 0, 0)
    cv2.putText(color_img, 'left curve: ' + str(round(left_curverad)) + ' m', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(color_img, 'right curve: ' + str(round(right_curverad)) + ' m', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(color_img, 'vehicle ' + str(round(lane_centre_drift*100)/100) + ' m from centre ', (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, thickness=2, lineType=cv2.LINE_AA)

    return color_img

images = glob.glob('./test_images/test*.jpg') + glob.glob('./test_images/straight_lines*.jpg')

for index, img_name in enumerate(images):
    lane_tracker = line.Line(margin=60, minpix=200)
    img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
    result = process_image(img)
    write_name = './test_images/track' + str(index+1) + '.jpg'
    cv2.imwrite(write_name, cv2.cvtColor(result,cv2.COLOR_RGB2BGR))    

output_video = 'project_video_tracked.mp4'
input_video = 'project_video.mp4'

lane_tracker = line.Line(margin=60, minpix=200)
clip = VideoFileClip(input_video)
output_clip = clip.fl_image(process_image)
output_clip.write_videofile(output_video, audio=False)
