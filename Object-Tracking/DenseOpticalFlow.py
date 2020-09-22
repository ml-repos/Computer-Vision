import cv2
import numpy as np

cap = cv2.VideoCapture('Video/chaplin.mp4')

ret, fframe = cap.read()
prev_gray = cv2.cvtColor(fframe, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(fframe)
mask[...,1]=255

while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('Input Frame',frame)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray,gray,None,0.5,3,15,3,5,1.2,0)

    magn, angel = cv2.cartToPolar(flow[...,0],flow[...,1])
    mask[...,0] = angel*180/np.pi/2
    mask[...,2] = cv2.normalize(magn,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)

    cv2.imshow('Dense Optical Flow',rgb)
    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()