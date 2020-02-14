import numpy    as np
import cv2      as cv

#cap = cv.VideoCapture(cv.samples.findFile('vtest.avi'))
#cap = cv.VideoCapture(cv.samples.findFile('slow_traffic_small.mp4'))
cap = cv.VideoCapture(0)

ret, frame_1 = cap.read()

# convert frame to grayscale & create mask
prvs        = cv.cvtColor(frame_1, cv.COLOR_BGR2GRAY)
hsv         = np.zeros_like(frame_1) # returns an array of zeros with the same shape as given array (frame_1)
hsv[..., 1] = 255

# window setup (allows window resizing and fullscreen)
screen_res      = 1920, 1080                                        # define screen resolution

scale_width     = screen_res[0] / frame_1.shape[1]                  #
scale_height    = screen_res[1] / frame_1.shape[0]                  # define scale
scale           = min(scale_width, scale_height)                    #

window_width    = int(frame_1.shape[1] * scale)                     # define resized window dimensions
window_height   = int(frame_1.shape[0] * scale)                     #

cv.namedWindow('Dense-Optical-Flow', cv.WINDOW_NORMAL)              # enable window resizing
cv.resizeWindow('Dense-Optical-Flow', window_width, window_height)  # resize according to screen

while(1):
    ret, frame_2    = cap.read()
    frame_2         = cv.flip(frame_2, 1)
    
    next = cv.cvtColor(frame_2, cv.COLOR_BGR2GRAY)

    # compute dense optical flow using the gunnar farneback algorithm - @params:
    # https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html
    # ---- [1]  prev......: first 8-bit single channel input image
    # ---- [2]  next......: second input image of the same size and type as 'prev'
    # ---- [3]  flow......: computed flow image that has the same size as 'prev' and type 'CV_32FC2'
    # ---- [4]  pyr_scale.: specifies image scale to build pyramids for each image (0.5 = classical pyramid)
    # ---- [5]  levels....: no. of pyramid layers including the initial image
    # ---- [6]  winsize...: average window size
    # ---- [7]  iterations: no. of iterations the algorithm does at each pyramid level
    # ---- [8]  poly_n....: size of the pixel neighborhood
    # ---- [9]  poly_sigma: standard deviation of the gaussian that is used to smooth derivatives
    # ---- [10] flags.....: operation flags
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # calculate magnitude and angle of 2d vectors
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # display window
    cv.imshow('Dense-Optical-Flow', bgr)

    # wait for user input to exit
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame_2)
        cv.imwrite('opticalhsv.png', bgr)
    
    # update previous frame
    prvs = next