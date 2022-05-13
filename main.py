import numpy as np
import cv2
import random
import math
from utils import onmouse_click, stereo_BM, stereo_SGM
import camera

WIN_NAME = 'Distance Measurement'
cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)


while True:


    # TODO - choose a image to measure, img: 1, 2, 3
    img = '1' # img: 1, 2, 3
    root = './data/'
    root_save = f'./result/{img}_5_subpixel_144/'
    disparity = np.load(f'{root_save}fillHole_median.npy')

    # TODO - please uncomment the following line for processing from beginning
    imgL = cv2.imread(f'{root}{img}_left.png', 0)
    imgR = cv2.imread(f'{root}{img}_right.png', 0)

    # rectify image using matrices obtained from calibration
    warp1 = cv2.remap(imgL, camera.left_map1, camera.left_map2, cv2.INTER_LINEAR)
    warp2 = cv2.remap(imgR, camera.right_map1, camera.right_map2, cv2.INTER_LINEAR)

    # parameters
    numDisparities = 144
    p1 = 10
    p2 = 150
    blockSize = 5
    algorithm = 'SGM'

    disparity, disparity_agg, disparity_left, census, dsi_agg = stereo_SGM(imgL=warp1, imgR=warp2, numDisparities=numDisparities, blockSize=blockSize, p1=p1, p2=p2)
    disparity = cv2.medianBlur(np.uint8(disparity), 5)


    # normalise to 0-255
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    threeD = cv2.reprojectImageTo3D(disp, camera.Q, handleMissingValues=True)
    threeD = threeD * 16

    cv2.setMouseCallback(WIN_NAME, onmouse_click, threeD)

    # display the disparity map
    cv2.imshow(WIN_NAME, disp)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break



cv2.destroyAllWindows()



