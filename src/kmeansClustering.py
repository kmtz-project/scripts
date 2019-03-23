import cv2
import numpy as np
 
class Segment:
    def __init__(self,segments=5):
        #define number of segments, with default 5
        self.segments=segments

    def kmeans(self,image):
       #Preprocessing step
       image=cv2.GaussianBlur(image,(7,7),0)
       vectorized=image.reshape(-1,3)
       vectorized=np.float32(vectorized)
       criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
              10, 1.0)
       ret,label,center=cv2.kmeans(vectorized,self.segments,None,
              criteria,10,cv2.KMEANS_RANDOM_CENTERS)
       res = center[label.flatten()]
       segmented_image = res.reshape((image.shape))
       return label.reshape((image.shape[0],image.shape[1])), segmented_image.astype(np.uint8)

seg_num = 5
base_dir    = "../data/pattern/"
img_path    = base_dir + "3.jpg"
result_path = base_dir + "3_kmeans.png"

image = cv2.imread(img_path)

seg           = Segment(seg_num)
label, result = seg.kmeans(image)

cv2.imshow("input", image)
cv2.imshow("segmented", result)
cv2.waitKey(0)

cv2.imwrite(result_path, result)