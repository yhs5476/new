import numpy as np
import cv2

# 주어진 넘파이 배열
data = np.array([[0,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0],
                [0,1,1,1,1,1,0,0],
                [0,1,1,0,1,1,1,0]], dtype=np.uint8)

img_data = data*255

cv2.imwrite('image.png', img_data)


kennel=cv2.getStructuringElement(cv2.MORPH_RECT,(1,2))

erode = cv2.erode(data, kennel)
dilate = cv2.dilate(data, kennel)

img_data1 = erode*255
img_data2 = dilate*255

cv2.imwrite('image1.png', img_data1)
cv2.imwrite('image2.png', img_data2)

