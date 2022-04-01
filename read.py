import cv2
from PIL import Image
import numpy as np
# txt_file = open('/home/zbc/Visual Computing/Assignment3-Py-Version/Assignment3-Py-Version/CMPT733-Lab3-Workspace/data/train/annotations/00000.txt')
# line = txt_file.readlines()
# print(line[0].split(' ')[1])
# image = cv2.imread('/home/zbc/Visual Computing/Assignment3-Py-Version/Assignment3-Py-Version/CMPT733-Lab3-Workspace/data/train/images/00002.jpg')
# image = cv2.resize(image, (20,40))
# print(image.shape[0])
# print(image.shape[1])
#
# cv2.imshow("image",image)
# cv2.waitKey(0)
#print(image)
# ious = [1,7,6,3]
# threshold = np.argmax(ious)
# print(threshold)
#ious_true = ious>threshold
#print(np.minimum(ious, 0.1))
#
# image = Image.open('/home/zbc/Visual Computing/Assignment3-Py-Version/Assignment3-Py-Version/CMPT733-Lab3-Workspace/data/train/images/00002.jpg')
# print(image.size)
# image = np.array(image)
# print(image)
a = np.zeros((3,3))
print(a)
a = np.zeros((2,5))
b = np.ones((2,3))
c = np.zeros((2,4))
print(np.concatenate((a,b,c),axis=1))
