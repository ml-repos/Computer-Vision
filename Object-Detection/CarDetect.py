import cv2

car_class = cv2.CascadeClassifier('object/Haarcascades/haarcascade_car.xml')
cap = cv2.VideoCapture('object/Video/Vehicles.mp4')

while cap.isOpened():
    success, frame = cap.read()
    cars = car_class.detectMultiScale(frame, 1.2, 3)
    if success:
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
            cv2.imshow("Cars", frame)
        if cv2.waitKey(1) & 0xff == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
