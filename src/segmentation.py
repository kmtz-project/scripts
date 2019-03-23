import cv2
import numpy as np

base_dir = "../data/pattern/"
img_filename = base_dir + "1.jpg"
dst_filename = base_dir + "1_segment.jpg"

image = cv2.imread(img_filename)

spatialRadius = 20
colorRadius   = 15
pyramidLevels = 3

dst = np.zeros((image.shape[0], image.shape[1]), np.ubyte)

dst = cv2.pyrMeanShiftFiltering(image, spatialRadius, colorRadius, pyramidLevels)
cv2.imshow("MeanShift", dst)
cv2.waitKey()

cv2.imwrite(dst_filename, dst)