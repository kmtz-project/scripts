import numpy as np
import cv2
from matplotlib import pyplot as plt

base_dir = "../data/pattern/"
img_filename = base_dir + "2.jpg"
dst_filename = base_dir + "2_denoise.jpg"

img = cv2.imread(img_filename)

dst = cv2.fastNlMeansDenoisingColored(img, 20, 7, 51)

cv2.imshow("input", img)
cv2.imshow("denoising", dst)
cv2.waitKey(0)

cv2.imwrite(dst_filename, dst)