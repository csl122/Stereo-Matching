from matplotlib import pyplot as plt
import camera
import cv2

# %% save rectified self-taken images for demonstration
images = range(1, 4)
for img in images:


    root = './data/'

    img1 = cv2.imread(f'{root}{img}_left.png', 0)
    img2 = cv2.imread(f'{root}{img}_right.png', 0)

    img1_rectified = cv2.remap(img1, camera.left_map1, camera.left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(img2, camera.right_map1, camera.right_map2, cv2.INTER_LINEAR)

    for i in range(int(img1.shape[0] / 48)):
        img1[i*48,:] = 255
        img2[i*48,:] = 255
        img1_rectified[i*48,:] = 255
        img2_rectified[i*48,:] = 255

    plt.subplot(2, 2, 1), plt.imshow(img1, cmap='gray')
    plt.subplot(2, 2, 2), plt.imshow(img2, cmap='gray')
    plt.subplot(2, 2, 3), plt.imshow(img1_rectified, cmap='gray')
    plt.subplot(2, 2, 4), plt.imshow(img2_rectified, cmap='gray')

    root_save = f'./result/{img}_5_subpixel_144/'
    plt.savefig(f'{root_save}rectify.png')

    plt.show()

# %% save images with holes filled for self-taken images
from utils import consistency_check, normalise
import numpy as np
from matplotlib import pyplot as plt
import cv2

images = range(1, 4)
for img in images:
    root_save = f'./result/{img}_5_subpixel_144/'

    dsi_agg = np.load(f'{root_save}dsi_agg.npy')
    disparity = consistency_check(dsi_agg, threshold=4, ifSubpixel=True, ifFillHole=True)
    disparity_median = cv2.medianBlur(np.uint8(disparity), 5)
    np.save(f'{root_save}fillHole.npy', disparity)
    cv2.imwrite(f'{root_save}fillHole_median.png', normalise(disparity_median))
    plt.imshow(disparity_median, 'gray')
    plt.show()

# %% save images with holes filled for public images
from utils import consistency_check, normalise
import numpy as np
from matplotlib import pyplot as plt
import cv2

images = ['bowling', 'cones',  'deer']
for img in images:
    root_save = f'./result/{img}_5_subpixel/'

    dsi_agg = np.load(f'{root_save}dsi_agg.npy')
    disparity = consistency_check(dsi_agg, threshold=4, ifSubpixel=True, ifFillHole=True)
    disparity_median = cv2.medianBlur(np.uint8(disparity), 5)
    np.save(f'{root_save}fillHole.npy', disparity)
    cv2.imwrite(f'{root_save}fillHole_median.png', normalise(disparity_median))
    plt.imshow(disparity_median, 'gray')
    plt.show()