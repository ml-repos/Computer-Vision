import cv2

body_class = cv2.CascadeClassifier('object/Haarcascades/haarcascade_fullbody.xml')

cap = cv2.VideoCapture('object/Video/People_Walking.mp4')

while cap.isOpened():
    success, frame = cap.read()
    bodies = body_class.detectMultiScale(frame, 1.2, 3)
    if success:
        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
            cv2.imshow("Perdestrains", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
