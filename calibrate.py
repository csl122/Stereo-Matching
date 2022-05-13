# %% calibration for left and right cameras
import cv2
import numpy as np
import glob

CHECKERBOARD = (7, 4)  # columns of checkerboard, rows of checkerboard
square_size = (35, 35)  # size of checkerboard square in mm
# termination criteria for checkerboard detection
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
img_points = []  # 2d points in image plane.
obj_points = []  # 3d points in real world space.
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
objp[:, 0] *= square_size[0]
objp[:, 1] *= square_size[1]

# TODO - choose a camera to calibrate, L or R
LR = 'R'

image_path = glob.glob(f'CALI{LR}/*.jpg')
i=0
for ip in image_path:
    img = cv2.imread(ip)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (7, 4), None)

    if ret:
        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        # uncomment to draw chessboard
        # cv2.drawChessboardCorners(img, (7, 4), corners, ret) 
        # cv2.imwrite(f'./CALI{LR}11/chessboard/CornerImg{i}.jpg', img)
        # i+=1
        

# get the camera matrix and distortion coefficients of the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

# print for reference
print("ret:", ret)
print("mtx:\n", mtx)
print("dist:\n", dist)
print("rvecs:\n", rvecs)
print("tvecs:\n", tvecs )


# save parameters in a dictionary
result = {}
result['ret'] = ret
result['mtx'] = mtx
result['dist'] = dist
result['rvecs'] = rvecs # rotation matrix in world coordinate
result['tvecs'] = tvecs # translation vector in world coordinate

# save parameters, please move the file to camera_parameters folder if confirmed
np.save(f'CALI{LR}/calibresult{LR}.npy', result)