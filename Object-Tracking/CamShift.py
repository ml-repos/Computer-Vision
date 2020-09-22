import cv2
import numpy as np

cap = cv2.VideoCapture('Video/face_track.mp4')

ret, frame = cap.read()
face_class = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
face_rect = face_class.detectMultiScale(frame)

fx,fy,w,h = tuple(face_rect[0])
track_window = (fx,fy,w,h)
roi = frame[fy:fy+h, fx:fx+w]
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([roi_hsv],[0],None,[180],[0,180])

cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
term_critaria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if ret:
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        dest_roi = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret, track_window = cv2.CamShift(dest_roi, track_window, term_critaria)
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img = cv2.polylines(frame,[pts],True,(0,255,0),5)
        cv2.imshow('FaceTracker', img)
        if cv2.waitKey(1) & 0xff == 27:
            break
    else:
        break
