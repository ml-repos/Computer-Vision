import cv2
import numpy as np

img = np.zeros((512, 512, 3), np.int8)


def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 100, (255, 0, 255), -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img, (x, y), 50, (0, 255, 255), -1)


cv2.namedWindow('My_Drawing')
cv2.setMouseCallback('My_Drawing', draw_circle)

while True:
    cv2.imshow('My_Drawing', img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cv2.destroyAllWindows()
