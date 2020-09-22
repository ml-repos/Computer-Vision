import cv2


def select_tracker():
    print("1.BOOSITNG\n2.MIL\n3.KCF\n4.TLD\n5.MEDIANFLOW\n6.GOTURN\n7.MOSSE\n8.CSRT\n")
    choice = input("\nEnter Choice:")
    if choice == '1':
        tracker = cv2.TrackerBoosting_create()
    elif choice == '2':
        tracker = cv2.TrackerMIL_create()
    elif choice == '3':
        tracker = cv2.TrackerKCF_create()
    elif choice == '4':
        tracker = cv2.TrackerTLD_create()
    elif choice == '5':
        tracker = cv2.TrackerMedianFlow_create()
    elif choice == '6':
        tracker = cv2.TrackerGOTURN_create()
    elif choice == '7':
        tracker = cv2.TrackerMOSSE_create()
    elif choice == '8':
        tracker = cv2.TrackerCSRT_create()
    return tracker


tracker = select_tracker()
tracker_name = str(tracker).split()[0][1:]
cap = cv2.VideoCapture('Video/Vehicles.mp4')
ret, frame = cap.read()
roi = cv2.selectROI(frame,False)
ret = tracker.init(frame,roi)

while True:
    ret, frame = cap.read()
    success, roi = tracker.update(frame)
    x,y,w,h = tuple(map(int,roi))
    if success:
        p1 = (x,y)
        p2 = (x+w, y+h)
        cv2.rectangle(frame,p1,p2,(255,125,25),3)
    else:
        cv2.putText(frame,'Fail to Detect',(100,200),cv2.FONT_HERSHEY_COMPLEX,1,(25,125,255),3)
    cv2.putText(frame, tracker_name, (20, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 3)
    cv2.imshow(tracker_name,frame)
    if cv2.waitKey(1) & 0xff==27:
        break

cap.release()
cv2.destroyAllWindows()
