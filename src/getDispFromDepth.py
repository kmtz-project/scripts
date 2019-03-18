# coding: utf8

import sys
import numpy as np
import cv2

def getDispFromDepth(inDepthFileName, outDispFileName):
    
    depthFile = open(inDepthFileName, "r")

    img_array = np.zeros((720, 1024), np.float32)

    num_lines = 0
    data_start_index = 24
    for line in depthFile:
        num_lines += 1
        val_array = line.split(",")
        if(num_lines >= data_start_index):
            idx = num_lines - data_start_index
            img_array[idx] = val_array[2:]
    
    # normalization
    img_array = img_array/img_array.max()*255

    print(img_array.max())
    print("Num lines:", num_lines)
    
    cv2.imwrite(outDispFileName, img_array)

# main
depthFileName = "../data/scene_v1_pt_left_dist.csv"
dispFileName  = "../data/disp.png"

getDispFromDepth(depthFileName, dispFileName)