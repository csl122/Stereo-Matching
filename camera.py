# %%
import cv2
import numpy as np

caliL = np.load('camera_parameters/calibresultL.npy', allow_pickle=True)
caliR = np.load('camera_parameters/calibresultR.npy', allow_pickle=True)
caliDual = np.load('camera_parameters/calibresultDual.npy', allow_pickle=True)

# intrinsic matrix of left camera
left_camera_matrix = caliL.item().get('mtx')
# distortion coefficients of left camera
left_distortion = caliL.item().get('dist')

# intrinsic matrix of right camera
right_camera_matrix = caliR.item().get('mtx')
# distortion coefficients of right camera
right_distortion = caliR.item().get('dist')

# rotation and transformation matrix of the right camera according to the left camera
R = caliDual.item().get('r')
T = caliDual.item().get('t')

# size of images
size = (640, 480)


# R1, R2 are the rotation matrices, P1, P2 are the projection matrices
# stereoRectify is used to calculate the rectification transformation matrices
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

# map1 and map2 are remap matrix for x and y axes
# initUndistortRectifyMap is used to calculate the remap matrix for x and y axes; used to rectify the images
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

