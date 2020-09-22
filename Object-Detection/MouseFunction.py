import cv2
import numpy as np

img = np.zeros((512, 512, 3), np.int8)

draw = False
ex, ey = -1, -1


def draw_rect(event, x, y, flags, param):
    global ex, ey, draw
    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
        ex, ey = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if draw:
            cv2.rectangle(img, (ex, ey), (x, y), (255, 0, 0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        draw = False
        cv2.rectangle(img, (ex, ey), (x, y), (255, 0, 0), -1)


cv2.namedWindow('My_Drawing')
cv2.setMouseCallback('My_Drawing', draw_rect)

while True:
    cv2.imshow('My_Drawing', img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cv2.destroyAllWindows()
