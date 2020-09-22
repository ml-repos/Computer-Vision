import numpy as np
import cv2

cap = cv2.VideoCapture('Video/run.mp4')

feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.2,
                       minDistance = 7,
                       blockSize = 7 )

lk_params = dict( winSize = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))

color = (0,255,0)

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
mask = np.zeros_like(old_frame)

while cap.isOpened():
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    good_new = p1[st==1]
    good_old = p0[st==1]

    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color, 2)
        frame = cv2.circle(frame,(a,b),3,color,-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xff == 27:
        break

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()






























































# import cv2
# import numpy as np
#
# st_param = dict(maxCorners=50, qualityLevel=0.2, minDistance=2, blockSize=7)
# lk_param = dict(winsize=(15, 15), maxDistance=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, -1))
#
# cap = cv2.VideoCapture('Video/run.mp4')
#
# success, fframe = cap.read()
# prev_gray = cv2.cvtColor(fframe, cv2.COLOR_BGR2GRAY)
# prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **st_param)
# mask = np.zeros_like(fframe)
#
# while cap.isOpened():
#     success, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_param)
#     good_old = prev[status == 1]
#     good_new = next[status == 1]
#     for i, (new, old) in enumerate(zip(good_new, good_old)):
#         a, b = new.ravel()
#         c, d = old.ravel()
#         mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
#         frame = cv2.circle(frame, (a, b), 3, (0, 255, 0), -1)
#     output = cv2.add(frame, mask)
#     prev_gray = gray.copy()
#     prev = good_new.reshape(-1, 1, 2)
#
#     cv2.imshow('Optical Flow', output)
#     if cv2.waitKey(1) & 0xff == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()
