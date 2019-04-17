import cv2
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass

cv2.namedWindow('res')

cv2.createTrackbar('max','res',0,255,nothing)
cv2.createTrackbar('min','res',0,255,nothing)

img = cv2.imread('1.jpg',0)
img = cv2.resize(img, (312,416))
maxVal=200
minVal=100

while (1):
    
    if cv2.waitKey(20) & 0xFF==27:
        break
    maxVal = cv2.getTrackbarPos('min','res')
    minVal = cv2.getTrackbarPos('max','res')
    if minVal < maxVal:
        edge = cv2.Canny(img,100,200)
        hmerge = np.hstack((img, edge)) #水平拼接
        cv2.imshow("res", hmerge) #拼接显示为gray
        # cv2.imshow('res',edge)
    else:
        edge = cv2.Canny(img,minVal,maxVal)
        hmerge = np.hstack((img, edge)) #水平拼接
        cv2.imshow("res", hmerge) #拼接显示为gray
        # cv2.imshow('res',edge)
cv2.destoryAllWindows()
