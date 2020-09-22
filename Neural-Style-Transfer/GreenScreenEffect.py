import cv2
import numpy as np

image = cv2.imread('./Image/car.jpg')
image_copy = image.copy()
image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)
lgreen = np.array([0,255,0])
ugreen = np.array([250,255,250])

mask = cv2.inRange(image_copy,lgreen,ugreen)
mask_img = image_copy.copy()
mask_img[mask!=0] = [0,0,0]

x,y,h,w = 390,300,450,660
bgimg = cv2.imread('./Image/london.jpg')
bgimg = cv2.cvtColor(bgimg,cv2.COLOR_BGR2RGB)
cbg = bgimg[y:y+h,x:x+w]
cbg[mask==0] = [0,0,0]

complete = mask_img + cbg

cv2.imshow('.',complete)
cv2.waitKey(0)