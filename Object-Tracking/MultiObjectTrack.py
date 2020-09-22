import cv2
import sys
from random import randint

tracker_type = ['BOOSITNG', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def tracker_name(tracker_type):
    if tracker_type == tracker_type[0]:
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == tracker_type[1]:
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == tracker_type[2]:
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == tracker_type[3]:
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == tracker_type[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == tracker_type[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == tracker_type[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif tracker_type == tracker_type[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('No Tracker Found')
    return tracker


if __name__ == '__main__':
    tracker_type = 'MOSSE'
    cap = cv2.VideoCapture('Video/Vehicles.mp4')
    success, frame = cap.read()
    rect, color = [], []
    while True:
        rect_box = cv2.selectROI('MultiTracker',frame)
        rect.append(rect_box)
        color.append((randint(64,255),randint(64,255),randint(64,255)))
        print('Press q to select box and start multiTrack')

        print('Press any key to select another box')
        if cv2.waitKey(1) & 0xff==113   :
            break
    print(f'selected Box{rect}')
    multitracaker = cv2.MultiTracker_create()
    for rect_box in rect:
        multitracaker.add(tracker_name(tracker_type),frame,rect_box)
    while cap.isOpened():
        success, frame = cap.read()
        success, boxes = multitracaker.update(frame)
        for i,newbox in enumerate(boxes):
            p1 = (int(newbox[0]),int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]),int(newbox[1] + newbox[3]))
            cv2.rectangle(frame,p1,p2,color[i],2,1)
        cv2.imshow('Multitracker',frame)
        if cv2.waitKey(1) & 0xff==27:
            break

cap.release()
cv2.destroyAllWindows()