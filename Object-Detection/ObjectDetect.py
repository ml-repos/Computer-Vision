import cv2

face_class = cv2.CascadeClassifier("object/Haarcascades/haarcascade_frontalface_default.xml")
eye_class = cv2.CascadeClassifier("object/Haarcascades/haarcascade_eye.xml")
image = cv2.imread('object/Images/eye_face.jpg')
fix_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
faces = face_class.detectMultiScale(image, 1.3, 5)
if faces is ():
    print('No Face Found')


def detect_face(fix_img):
    face_rect = face_class.detectMultiScale(fix_img)
    for (x, y, w, h) in face_rect:
        cv2.rectangle(fix_img, (x, y), (x + w, y + h), (255, 0, 0), 10)

    eye_rect = eye_class.detectMultiScale(fix_img)
    for (ix, iy, iw, ih) in eye_rect:
        cv2.rectangle(fix_img, (ix, iy), (ix + iw, iy + ih), (255, 0, 255), 5)
    return fix_img


result = detect_face(fix_img)

show = cv2.resize(result, None, fx=0.25, fy=0.25)
cv2.imshow('Result', show)
cv2.waitKey()
