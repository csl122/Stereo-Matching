import numpy as np
import math
import os
import cv2
import math
from matplotlib import pyplot as plt


def matching_cost(img1, img2, numDisparities, window_size, algorithm):
    """
    Compute the matching cost between two images
    :param algorithm: algorithm for matching cost
    :param img1: grayscale image
    :param img2: grayscale image
    :param numDisparities: number of disparity
    :param window_size: size of the window
    :return: disparity_local_left, dsi_left, census_map
    """

    assert algorithm in ['SAD', 'SSD', 'NCC', 'census'], 'Algorithm not supported'
    h, w = img1.shape
    # Disparity map obtained by best costs
    disparity_local_left = np.zeros((h, w))
    # Disparity Space Image
    dsi_left = np.full((h, w, numDisparities), 65535.)
    half_window_size = window_size // 2
    census_map=np.zeros((h, w))

    for i in range(half_window_size, h-half_window_size):
        for j in range(half_window_size, w-half_window_size):
            # original images, looking for a match
            origin = img1[i-half_window_size:i+half_window_size+1, j-half_window_size:j+half_window_size+1]
            # initialise best cost for Sum of Absolute Differences, Sum of Squared Differences and Normalized Cross Correlation
            best = 0 if algorithm == 'NCC' else 65535.
            census_map[i,j]=np.sum(origin < origin[half_window_size, half_window_size])
            disparity = 0
            for d in range(0, min(numDisparities, j-half_window_size)):
                target = img2[i-half_window_size:i+half_window_size+1, j-half_window_size-d:j+half_window_size-d+1]
                if algorithm == 'SAD':
                    cost = np.sum(np.abs(origin - target))
                elif algorithm == 'SSD':
                    cost = np.sum(np.square(origin - target))
                elif algorithm == 'NCC':
                    origin_norm = (origin - np.mean(origin)) / np.std(origin)
                    target_norm = (target - np.mean(target)) / np.std(target)
                    cost = np.sum(np.multiply(origin_norm, target_norm))
                elif algorithm == 'census':
                    compare_left = origin < origin[half_window_size, half_window_size]
                    compare_right = target < target[half_window_size, half_window_size]
                    cost = np.sum(np.bitwise_xor(compare_left, compare_right))

                if cost < best and algorithm != 'NCC':
                    best = cost
                    disparity = d
                elif cost > best and algorithm == 'NCC':
                    best = cost
                    disparity = d


                # Form the Disparity Space Image
                dsi_left[i, j, d] = cost

            # Disparity value with best cost for pixel (i, j)
            disparity_local_left[i, j] = disparity


    return disparity_local_left, dsi_left, census_map


def normalise(img):
    """
    Normalise the image to 0-255 range
    :param img:
    :return: normalised image
    """
    return (img - np.min(img)) / (np.max(img) - np.min(img)) * 255


def sobel(img):
    """
    Compute the sobel operator on the image
    :param img: input image
    :return: image after sobel filter
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    out = (sobelx ** 2 + sobely ** 2) ** 0.5

    return out


def pre_process(img):
    """
    Pre-process the image to get rid of noise and to get rid of the noise in the edges
    :param img: input image
    :return: processed image
    """
    # Convert to grayscale
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    # img_bila = cv2.bilateralFilter(img,3,10,10)
    # Apply Sobel Filter
    img_sobel = sobel(img_blur)

    # Apply Threshold
    # img_thresh = cv2.threshold(img_sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return img_sobel


def CostAggregateLeftRight(img, dsi, numDisparities, p1, p2, isFromLeft):
    """
    Aggregate the costs for the left and right images
    :param dsi: disparity space image
    :param numDisparities: number of disparities
    :param p1: parameter for the aggregation
    :param p2: parameter for the aggregation
    :return: aggregated disparity map
    """
    h, w = dsi.shape[0], dsi.shape[1]

    disparity_map = np.zeros((h, w))
    cost_aggr = np.zeros(dsi.shape)

    for i in range(h):
        start = 0 if isFromLeft else w-1
        gray = img[i, start]
        gray_last = img[i, start]

        cost_last_path = np.full(numDisparities+2, 65535.)

        cost_aggr[i, start, :] = dsi[i, start, :]
        cost_last_path[1:-1] = dsi[i, start, :]

        mincost_last_path = np.min(cost_last_path)
        disparity_map[i, start] = np.argmin(cost_last_path) - 1

        for k in range(1, w):
            j = k if isFromLeft else w-k-1
            gray = img[i, j]
            min_cost = 65535.
            for d in range(0, numDisparities):
                cost = dsi[i, j, d]
                l1 = cost_last_path[d + 1]
                l2 = cost_last_path[d] + p1
                l3 = cost_last_path[d + 2] + p1
                l4 = mincost_last_path + max(p1, p2 / (1 + abs(int(gray) - int(gray_last))))

                cost_s = cost + min(l1, l2, l3, l4) - mincost_last_path
                cost_aggr[i, j, d] = cost_s
                min_cost = min(min_cost, cost_s)

            mincost_last_path = min_cost
            cost_last_path[1:-1] = cost_aggr[i, j, :]
            disparity_map[i, j] = np.argmin(cost_last_path) - 1

            gray_last = gray

    return cost_aggr, disparity_map


def CostAggregateUpDown(img, dsi, numDisparities, p1, p2, isFromUp):
    """
    Aggregate the costs for the left and right images
    :param dsi: disparity space image
    :param numDisparities: number of disparities
    :param p1: parameter for the aggregation
    :param p2: parameter for the aggregation
    :return: aggregated disparity map
    """
    h, w = dsi.shape[0], dsi.shape[1]
    direction = 1 if isFromUp else -1
    disparity_map = np.zeros((h, w))
    cost_aggr = np.zeros(dsi.shape)

    for i in range(w):
        start = 0 if isFromUp else h-1
        gray = img[start, i]
        gray_last = img[start, i]

        cost_last_path = np.full(numDisparities + 2, 65535.)

        cost_aggr[start, i, :] = dsi[start, i, :]
        cost_last_path[1:-1] = dsi[start, i, :]

        mincost_last_path = np.min(cost_last_path)
        disparity_map[start, i] = np.argmin(cost_last_path) - 1

        for k in range(1, h):
            j = k if isFromUp else h - k - 1
            gray = img[j, i]
            min_cost = 65535.
            for d in range(0, numDisparities):
                cost = dsi[j, i, d]
                l1 = cost_last_path[d + 1]
                l2 = cost_last_path[d] + p1
                l3 = cost_last_path[d + 2] + p1
                l4 = mincost_last_path + max(p1, p2 / (1 + abs(int(gray) - int(gray_last))))

                cost_s = cost + min(l1, l2, l3, l4) - mincost_last_path
                cost_aggr[j, i, d] = cost_s
                min_cost = min(min_cost, cost_s)

            mincost_last_path = min_cost
            cost_last_path[1:-1] = cost_aggr[j, i, :]
            disparity_map[j, i] = np.argmin(cost_last_path) - 1

            gray_last = gray

    return cost_aggr, disparity_map


def CostAggregate(img, dsi, numDisparities, p1, p2):

    dsi_lr, disparity = CostAggregateLeftRight(img, dsi, numDisparities, p1, p2, True)
    print('Finish left->right aggregation.')

    dsi_rl, disparity = CostAggregateLeftRight(img, dsi, numDisparities, p1, p2, False)
    print('Finish right->left aggregation.')

    dsi_ud, disparity = CostAggregateUpDown(img, dsi, numDisparities, p1, p2, True)
    print('Finish up->down aggregation.')

    dsi_du, disparity = CostAggregateUpDown(img, dsi, numDisparities, p1, p2, False)
    print('Finish down->up aggregation.')

    dsi_agg = dsi_lr + dsi_rl + dsi_ud + dsi_du
    disparity = np.argmin(dsi_agg, 2)

    return dsi_agg, disparity
    
    
# use winner takes all to compute disparity map
def wta(dsi):
    """
    Use winner takes all to compute disparity map
    :param dsi: disparity space image
    :return: aggregated disparity map
    """
    disparity = np.argmin(dsi, 2)
    return disparity


def subpixel(dsi):
    """
    Use subpixel to compute disparity map
    :param dsi: disparity space image
    :return: aggregated disparity map
    """
    h, w, d = dsi.shape
    disparity = np.zeros(dsi.shape[:2])
    for i in range(h):
        for j in range(w):
            min = np.argmin(dsi[i, j, :])
            if min == 0 or min == d - 1:
                disparity[i, j] = min
            else:
                temp = max(1, dsi[i, j, min + 1] + dsi[i, j, min - 1] - 2 * dsi[i, j, min])
                disparity[i, j] = min + (dsi[i, j, min - 1] - dsi[i, j, min + 1]) / (2 * temp)
    return disparity


def check_disparity(disp1, disp2, threshold = 4):
    """
    Check if the disparity map is correct
    :param threshold: threshold for left right consistency
    :param disp1: disparity map
    :param disp2: disparity map
    :return: checked disparity map
    """
    h, w = disp1.shape
    disparity_map = np.zeros((h, w))
    occlusion_map = []
    mismatch_map = []
    for i in range(h):
        for j in range(w):
            left_disp = np.int64(disp1[i, j]+0.5)
            right_disp = disp2[i, j-left_disp]

            if abs(disp1[i, j] - right_disp) > threshold:
                if j-left_disp+right_disp > 0 and j-left_disp+right_disp < w:
                    d_prime = disp1[i, j - left_disp + np.int64(right_disp+0.5)]
                    if d_prime > disp1[i, j]:
                        occlusion_map.append((i, j))
                    else:
                        mismatch_map.append((i, j))
                else:
                    mismatch_map.append((i, j))
                disparity_map[i, j] = 0
            else:
                disparity_map[i, j] = disp1[i, j]

    return disparity_map, occlusion_map, mismatch_map

def fill_hole(disparity, occlusion_map, mismatch_map):
    h, w = disparity.shape
    disparity_map = disparity.copy()
    pi = math.pi
    angles = [pi, 3 * pi / 4, pi / 2, pi / 4, 0, 7 * pi / 4, 3 * pi / 2, 5 * pi / 4]
    print(len(occlusion_map), len(mismatch_map))
    for k in range(3):
        map = occlusion_map if k == 0 else mismatch_map
        if k == 2:
            map = []
            for i in range(h):
                for j in range(w):
                    if disparity[i, j] == 0:
                        map.append((i, j))

        for i in map:
            y = i[0]
            x = i[1]

            valid_disp= []
            for angle in angles:
                sina = math.sin(angle)
                cosa = math.cos(angle)
                for n in range(h+w):
                    y_prime = np.int64(y+n*sina)
                    x_prime = np.int64(x+n*cosa)
                    if y_prime<0 or y_prime >= h or x_prime<0 or x_prime >= w:
                        break
                    disp = disparity[y_prime, x_prime]
                    if disp > 0:
                        valid_disp.append(disp)
                        break

            if not valid_disp:
                continue

            valid_disp.sort()

            # if k == 0:
            #     if len(valid_disp) > 1:
            #         disparity[y, x] = valid_disp[1]
            #         # print(valid_disp)
            #     else:
            #         disparity[y, x] = valid_disp[0]
            # else:
            disparity[y, x] = valid_disp[len(valid_disp) // 2]

    return disparity



def consistency_check(dsi, threshold, ifSubpixel, ifFillHole):
    '''
    Check if the two left and right disparity maps are matched, black pixel means no match
    :param dsi: disparity space image
    :param threshold: threshold for disparity consistency
    :param ifSubpixel: whether use subpixel
    :return: checked disparity map
    '''

    h, w, d = dsi.shape
    dsi_right = np.full((h, w, d), 65535.)
    disparity_left = subpixel(dsi) if ifSubpixel else wta(dsi)

    for i in range(h):
        for j in range(w):
            for k in range(d):
                col_left = j+k
                if col_left < w:
                    a= dsi[i, col_left, k]
                    dsi_right[i, j, k] = dsi[i, col_left, k]
                    b=dsi_right[i, j, k]

    disparity_right = subpixel(dsi_right) if ifSubpixel else wta(dsi_right)
    disparity, occlusion_map, mismatch_map = check_disparity(disparity_left, disparity_right, threshold)
    disparity_median = cv2.medianBlur(np.uint8(disparity), 5)

    if ifFillHole:
        disparity = fill_hole(disparity_median.copy(), occlusion_map, mismatch_map)
    return disparity

def stereo_BM(imgL, imgR, numDisparities, blockSize, algorithm = 'SAD'):
    """
    Compute disparity map using block matching algorithm
    :param imgL: left image
    :param imgR: right image
    :param numDisparities: number of disparities
    :param blockSize: block size
    :return: disparity map
    """
    disparity_left, dsi_left, census_map = matching_cost(imgL, imgR, numDisparities, blockSize, algorithm)

    return disparity_left

def stereo_SGM(imgL, imgR, numDisparities, blockSize, p1, p2):
    """
    Compute disparity map using Semi-Global Matching algorithm
    :param imgL: left image
    :param imgR: right image
    :param numDisparities: number of disparities
    :param blockSize: block size
    :param p1: penalty for disparity change == 1
    :param p2: penalty for disparity change > 1
    :return: disparity, disparity_agg, disparity_left, census_map, dsi_agg
    """
    # calculate DSI
    disparity_left, dsi_left, census_map = matching_cost(imgL, imgR, numDisparities, blockSize, 'census')

    print('Finish DSI calculating.')

    # cost aggregation
    dsi_agg, disparity_agg = CostAggregate(imgL, dsi_left, numDisparities, p1, p2)

    print('Finish cost aggregation.')
    # dsi_agg = np.load('dsi_agg_cones.npy')
    # LR check and refinement
    disparity = consistency_check(dsi_agg, threshold=4, ifSubpixel=True, ifFillHole=True)
    print('Finish consistency check.')


    return disparity, disparity_agg, disparity_left, census_map, dsi_agg

def onmouse_click(event, x, y, flags, param):
    """
    Callback function for mouse click
    :param event: mouse event
    :param x: x coordinate
    :param y: y coordinate
    :param flags: //
    :param param: //
    :return: //
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        threeD = param
        print(f'Pixel chosen is at ({x}, {y})')
        print(f'3D position is at ({threeD[y, x, 0]/1000:.2f}, {threeD[y, x, 1]/1000:.2f}, {threeD[y, x, 2]/1000:.2f})')
        distance = math.sqrt(threeD[y][x][0] ** 2 + threeD[y][x][1] ** 2 + threeD[y][x][2] ** 2)/100
        print(f'Distance to camera is {distance:.2f} cm')


