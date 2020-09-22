import cv2
import numpy as np


def neural_style_transfer(img, model, size=320, upscale=1):
    model_file_path = './model/'
    style = cv2.imread('./art/' + str(model)[:-3] + '.jpg')
    neural_model = cv2.dnn.readNetFromTorch(model_file_path + model + '.t7')

    h, w = int(img.shape[0]), int(img.shape[1])
    nw = int((size / h) * w)
    resize_img = cv2.resize(img, (nw, size), interpolation=cv2.INTER_AREA)

    inp_blob = cv2.dnn.blobFromImage(resize_img, 1.0, (nw, size), (103.93, 116.77, 123.68), swapRB=False, crop=False)
    neural_model.setInput(inp_blob)
    output = neural_model.forward()

    output = output.reshape(3, output.shape[2], output.shape[3])
    output[0] += 103.93
    output[1] += 116.77
    output[2] += 123.68
    output /= 255
    output = output.transpose(1, 2, 0)
    output = cv2.resize(output, None, fx=upscale,fy=upscale, interpolation=cv2.INTER_LINEAR)
    return output


# cap = cv2.VideoCapture('./Video/run.mp4')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('Neural Transfer Style',neural_style_transfer(frame,'the_scream',320,2))
    if cv2.waitKey(1) & 0xff==27:
        break

cap.release()
cv2.destroyAllWindows()