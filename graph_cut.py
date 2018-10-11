import random
import numpy as np
import cv2
from scipy.io import loadmat



segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.5, k=200, min_size=3000)
src = cv2.imread('./BSR/BSDS500/data/images/test/100039.jpg')
segment = segmentator.processImage(src)
seg_image = np.zeros(src.shape, np.uint8)

for i in range(np.max(segment)):

  y, x = np.where(segment == i)

  color = [random.randint(0, 255), random.randint(0, 255),random.randint(0, 255)]


  for xi, yi in zip(x, y):
    seg_image[yi, xi] = color


result = cv2.addWeighted(src, 0.4, seg_image, 0.6, 0)
cv2.imwrite("/Users/zhongyiqi/Documents/ALI270/Midterm_Planning_Form/slides_code/graph_cun.jpg",result)
cv2.imwrite("/Users/zhongyiqi/Documents/ALI270/Midterm_Planning_Form/slides_code/source.jpg",src)


cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
