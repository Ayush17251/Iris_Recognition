import cv2
import numpy as np
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
img = cv2.imread("pic00020.jpg",0)
#cv2.imshow('img',img)
#cv2.waitKey()
crop_img = img[10:500, 240:285]
#hist=cv2.calcHist([crop_img],[0],None,[256],[0,256])
#plt.hist(crop_img.ravel(),256,[0,256]); plt.show()
#histeq = cv2.equalizeHist(crop_img)
#plt.hist(histeq.ravel(),256,[0,256]); plt.show()
#contrast=ImageEnhance.Contrast(crop_img)
#contrast.enhance(2).show()
a=np.rot90(crop_img,k=3)
cv2.imshow('img',a)
cv2.waitKey()

# Contrast Limited Adaptive Histogram Equalization

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1=clahe.apply(a)
cv2.imshow('img1',cl1)
cv2.waitKey()
cv2.imwrite('clahe_2.jpg',cl1)


