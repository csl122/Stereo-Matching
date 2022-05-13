# %% stereoBM algorithm
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import stereo_SGM, stereo_BM, normalise, pre_process
import os
import camera

# images = ['bowling', 'cones',  'deer']
images = range(1, 4)
for img in images:


    root = './data/'
    # img: bowling, cones, deer
    # img = '1'

    img1 = cv2.imread(f'{root}{img}_left.png', 0)
    img2 = cv2.imread(f'{root}{img}_right.png', 0)

    assert img1 is not None and img2 is not None, 'Image not found'
    assert img1.shape == img2.shape, "Images must have the same size"

    if img1.shape[1] > 640:
        size = (640, int(img1.shape[0] * 500 / img1.shape[1]))
        img1 = cv2.resize(img1, size)
        img2 = cv2.resize(img2, size)

    warp1 = img1
    warp2 = img2
    warp1 = pre_process(warp1)
    warp2 = pre_process(warp2)

    # TODO - please comment out the following lines for images from public dataset
    warp1 = cv2.remap(img1, camera.left_map1, camera.left_map2, cv2.INTER_LINEAR)
    warp2 = cv2.remap(img2, camera.right_map1, camera.right_map2, cv2.INTER_LINEAR)



    root_save = f'./result_bm/{img}_21/'
    if os.path.exists(root_save) is False:
        os.makedirs(root_save)

    numDisparities = 144
    # TODO - please uncomment the following line for images from public dataset
    numDisparities = ((warp1.shape[1] // 8) + 15) & -16

    p1 = 10
    p2 = 150
    blockSize = 21
    algorithm = 'BM'
    algorithm_cost = 'SAD'

    # for algorithm in ['SGM', 'BM']:
    start = time.perf_counter()
    if algorithm == 'SGM':
        disparity, disparity_agg, disparity_left, census, dsi_agg = stereo_SGM(imgL=warp1, imgR=warp2, numDisparities=numDisparities, blockSize=blockSize, p1=p1, p2=p2)
        cv2.imwrite(f'{root_save}disparity_SGM.png', normalise(disparity))
        cv2.imwrite(f'{root_save}disparity_SGM_agg.png', normalise(disparity_agg))
        np.save(f'{root_save}dsi_agg.npy', dsi_agg)
        cv2.imwrite(f'{root_save}disparity_SGM_noagg.png', normalise(disparity_left))
        cv2.imwrite(f'{root_save}census_SGM.png', normalise(census))
    elif algorithm == 'BM':
        disparity = stereo_BM(warp1, warp2, numDisparities, blockSize, algorithm_cost)
        cv2.imwrite(f'{root_save}disparity_BM_{algorithm_cost}.png', disparity)

    # post process
    disparity_median = cv2.medianBlur(np.uint8(disparity), 3)
    cv2.imwrite(f'{root_save}disparity_{algorithm}_median.png', normalise(disparity_median))
    np.save(f'{root_save}disparity_{algorithm}_median.npy', disparity_median)
    end = time.perf_counter()
    print(end - start)
    plt.imshow(disparity_median,'gray')
    plt.show()


