import cv2

img = cv2.imread('object/Images/car_plate3.jpg')
car_class = cv2.CascadeClassifier('object/Haarcascades/haarcascade_plate_number.xml')


def display(img):
    fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fix_img = cv2.resize(fix_img, None, fx=0.5, fy=0.5)
    cv2.imshow("Cars", fix_img)


def detect_plate(img):
    plate_rect = car_class.detectMultiScale(img, 1.1, 1)
    for (x, y, w, h) in plate_rect:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 10)
    return


result = detect_plate(img)

display(img)
cv2.waitKey()
