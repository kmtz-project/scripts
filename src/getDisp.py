# coding: utf8

import cv2
import numpy as np

# disparity settings
window_size = 3
min_disp = 0
num_disp = 96
stereo = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 2,
    uniquenessRatio = 15,
    speckleWindowSize = 0,
    speckleRange = 2,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

base_dir = "../data/scene_snowflake_1/"

image_left_filename  = base_dir + "im0.png"
image_right_filename = base_dir + "im1.png"
outDispFileName      = base_dir + "disp.png"
outDispFiltFileName  = base_dir + "disp_filt.png"

image_left = cv2.imread(image_left_filename)
image_right = cv2.imread(image_right_filename)

# compute disparity
disparity = stereo.compute(image_left, image_right).astype(np.float32) / 16.0

print(disparity.max())

disparity_out = (disparity)/num_disp * 256
cv2.imwrite(outDispFileName, disparity_out)

# Filtration
left_matcher  = stereo 
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
imgL = image_left
imgR = image_right

lmbda = 80000 # 80000
sigma = 0.7 # 1.2
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

print('computing disparity...')
displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
displ = np.int16(displ)
dispr = np.int16(dispr)
filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
filteredImg = np.uint8(filteredImg)
cv2.imwrite(outDispFiltFileName, filteredImg)



