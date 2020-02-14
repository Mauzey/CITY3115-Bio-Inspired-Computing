import numpy    as np
import cv2      as cv
import argparse

#cap = cv.VideoCapture('slow_traffic_small.mp4')
#cap = cv.VideoCapture('vtest.avi')
cap = cv.VideoCapture(0)

# @params for shitomasi corner detection
feature_params = dict(  maxCorners      = 100,  # max. no. of corners to return (if more are found, the strongest are returned)
                        qualityLevel    = 0.3,  # minimal accepted quality of image corners
                        minDistance     = 7,    # min. possible euclidean distance between returned corners
                        blockSize       = 7)    # size of average block for computing a derivative covariation matrix over each pixel neighborhood

# @params for lucas kanade optical flow
lk_params = dict(   winSize     = (15,15),                  # size of the search window at each pyramid level
                    maxLevel    = 2,                        # 0-based maximal pyramid level number
                    criteria    = (cv.TERM_CRITERIA_EPS |   # termination criteria
                    cv.TERM_CRITERIA_COUNT, 10, 0.03))

# create random colours for the points
color = np.random.randint(0, 255, (100,3))

# get first frame
ret, old_frame  = cap.read()
old_gray        = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# determine strong corners in the frame - @params:
# https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
# ---- [1] image...........: input 8-bit or floating-point 32-bit, single channel image
# ---- [2] mask............: optional region of interest - specifies region in which corners are detected
# ---- [3] **feature_params: dictionary containing further @params
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# create mask for drawing purposes
mask = np.zeros_like(old_frame) # returns an array of zeros with the same shape as given array (old_frame)

# window setup (allows window resizing and fullscreen)
screen_res      = 1920, 1080                                        # define screen resolution

scale_width     = screen_res[0] / old_frame.shape[1]                #
scale_height    = screen_res[1] / old_frame.shape[0]                # define scale
scale           = min(scale_width, scale_height)                    #

window_width    = int(old_frame.shape[1] * scale)                   # define resized window dimensions
window_height   = int(old_frame.shape[0] * scale)                   #

cv.namedWindow('Sparse-Optical-Flow', cv.WINDOW_NORMAL)             # enable window resizing
cv.resizeWindow('Sparse-Optical-Flow', window_width, window_height) # resize according to screen

while(1):
    ret, frame  = cap.read()
    frame       = cv.flip(frame, 1) # frame horizontally for natural movement
    frame_gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow - @params:
    # https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html
    # ---- [1] prevImg....: first 8-bit image or pyramid constructed by buildOpticalFlowPyramid()
    # ---- [2] nextImg....: second input image or pyramid of the same size and type as 'prevImg'
    # ---- [3] prevPts....: vector of 2D points for which the flow needs to be found
    # ---- [4] nextPts....: output vector of 2D points containing calculated positions of input features in the second image
    # ---- [5] **lk_params: dictionary containing further @params
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel() # ravel() returns a adjacent flattened array
        c, d = old.ravel()

        # draw points and trails
        mask    = cv.line(mask, (a,b), (c,d), color[i].tolist(), 2)
        frame   = cv.circle(frame, (a,b), 5, color[i].tolist(), -1)
    
    img = cv.add(frame, mask)

    # display window
    cv.imshow('Sparse-Optical-Flow', img)
    
    # wait for user input to exit
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # update previous frame and points
    old_gray    = frame_gray.copy()
    p0          = good_new.reshape(-1, 1, 2)