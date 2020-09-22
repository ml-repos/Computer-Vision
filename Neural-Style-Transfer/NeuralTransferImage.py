import cv2
from os.path import isfile,join
from os import listdir

model_file_path = './model/'
model_file_paths = [f for f in listdir(model_file_path) if isfile(join(model_file_path, f))]

img = cv2.imread('./Image/car.jpg')
model = ('udnie.t7')
for i in model_file_paths:
    style = cv2.imread('./art/' + str(model)[:-3] + '.jpg')

    neural_model = cv2.dnn.readNetFromTorch(model_file_path + model)
    h, w = int(img.shape[0]), int(img.shape[1])
    nw = int((640/h)*w)
    resize_img = cv2.resize(img,(nw,640),interpolation=cv2.INTER_AREA)

    inp_blob = cv2.dnn.blobFromImage(resize_img,1.0,(nw,640),(103.93,116.77,123.68),swapRB=False,crop=False)
    neural_model.setInput(inp_blob)
    output = neural_model.forward()

    output = output.reshape(3, output.shape[2],output.shape[3])
    output[0] += 103.93
    output[1] += 116.77
    output[2] += 123.68
    output /= 255
    output = output.transpose(1,2,0)

    # cv2.imshow('Original',img)
    # cv2.imshow('Style', style)
    cv2.imshow('Neural Style Transfer', output)
    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xff==27:
        break
cv2.destroyAllWindows()