# %% calibration for binocular camera
import cv2
import os
import numpy as np
import glob

left_path = 'CALI/left'
right_path = 'CALI/right'
CHECKERBOARD = (7, 4)  # columns of checkerboard, rows of checkerboard
square_size = (35, 35)  # size of checkerboard square in mm
# termination criteria for checkerboard detection
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
img_points_l = []  # 2d points in left image plane.
img_points_r = []  # 2d points in right image plane.
obj_points = []  # 3d points in real world space.
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp[0, :, 0] *= square_size[0]
objp[0, :, 1] *= square_size[1]

if os.path.exists('CALI/right') is False:
    os.makedirs('CALI/right')
    os.makedirs('CALI/left')
if os.path.exists('CALI/chessboard') is False:
    os.makedirs('CALI/chessboard')

# split the dual camera images into left and right
if os.path.exists('CALI/right/1.jpg') is False:
    images = glob.glob("CALI/images_dual/*.jpg")
    counter = 0
    for i in images:
        img = cv2.imread(i, 0)
        h, w = img.shape[:2]
        imgL = img[:,:int(img.shape[1]/2)]
        imgR = img[:,int(img.shape[1]/2):]
        cv2.imwrite(f'CALI/left/{counter}.jpg', imgL)
        cv2.imwrite(f'CALI/right/{counter}.jpg', imgR)
        counter += 1


# %%
i = 0
for ip in os.listdir(left_path):
    imgL = cv2.imread(os.path.join(left_path, ip), 0)
    imgR = cv2.imread(os.path.join(right_path, ip), 0)

    # find chess board corners
    ret_l, corners_l = cv2.findChessboardCorners(imgL, CHECKERBOARD)
    ret_r, corners_r = cv2.findChessboardCorners(imgR, CHECKERBOARD)

    # if found, add object points, image points (after refining them)
    if ret_l and ret_r:
        obj_points.append(objp)
        corners2_l = cv2.cornerSubPix(imgL, corners_l, (11, 11), (-1, -1), criteria)
        img_points_l.append(corners2_l)
        corners2_r = cv2.cornerSubPix(imgR, corners_r, (11, 11), (-1, -1), criteria)
        img_points_r.append(corners2_r)
        # Draw and display the corners
        # img = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
        # img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2_l, ret_l)
        # cv2.imwrite(f'./CALI/chessboard/CornerImg{i}.jpg', img)
    i += 1
# get the camera matrix and distortion coefficients of both cameras
ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(obj_points, img_points_l, imgL.shape[::-1], None, None)
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(obj_points, img_points_r, imgR.shape[::-1], None, None)

# calibrate stereo camera using their own calibration parameters
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(obj_points, img_points_l,
                                                                                                 img_points_r, mtx_l,
                                                                                                 dist_l, mtx_r, dist_r,
                                                                                                 imgL.shape[::-1])

# print for reference
print("ret_l: ", ret_l)
print("ret_r: ", ret_r)

print("mtx_l:\n", mtx_l)
print("dist_l:\n", dist_l)

print("mtx_r:\n", mtx_r)
print("dist_r:\n", dist_r)

print("R:\n", R)
print("T:\n", T)
print("E:\n", E)
print("F:\n", F)

# save parameters in a dictionary
result = {}
result['mtx_l'] = cameraMatrix1
result['mtx_r'] = cameraMatrix2
result['dist_l'] = distCoeffs1
result['dist_r'] = distCoeffs2
result['r'] = R  # rotation matrix for rotating the right camera coordinate to the left camera coordinate
result['t'] = T  # translation vector for translating the right camera coordinate to the left camera coordinate
result['e'] = E  # essential matrix
result['f'] = F  # fundamental matrix

# save parameters, please move the file to camera_parameters folder if confirmed
np.save('CALI/calibresultDual.npy', result)
