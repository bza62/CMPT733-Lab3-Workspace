import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import cv2
import torch.utils.data as data
#generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #TODO:
    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.
    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bSharif University of Technologyounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    number_of_boxes = 0
    number_of_layer = len(layers)
    for grid in layers:
        number_of_boxes += grid*grid

    # print("default box ")
    boxes = np.zeros([number_of_boxes,4,8])
    box_no = 0
    for i in range(0,number_of_layer):
        for row in range(0,layers[i]):
            for col in range(0,layers[i]):
                x_center = (col+0.5)/layers[i]
                y_center = (row+0.5)/layers[i]
                if layers[i]==3:
                    x_center = round(x_center,2)
                    y_center = round(y_center,2)
                l_s_scale = round(large_scale[i]/1.4,2)
                boxes[box_no,0] = [x_center,y_center,small_scale[i],small_scale[i],x_center-small_scale[i]/2,y_center-small_scale[i]/2,x_center+small_scale[i]/2,y_center+small_scale[i]/2]

                boxes[box_no,1] =[x_center,y_center,large_scale[i],large_scale[i],x_center-large_scale[i]/2,y_center-large_scale[i]/2,x_center+large_scale[i]/2,y_center+large_scale[i]/2]

                boxes[box_no,2] =[x_center,y_center,large_scale[i]*1.4,l_s_scale,x_center-large_scale[i]*1.4/2,y_center-l_s_scale/2,x_center+large_scale[i]*1.4/2,y_center+l_s_scale/2]

                boxes[box_no,3] =[x_center,y_center,l_s_scale,large_scale[i]*1.4,x_center-l_s_scale/2,y_center-large_scale[i]*1.4/2,x_center+l_s_scale/2,y_center+large_scale[i]*1.4/2]
                box_no += 1
                # print('default box in row %d col %d in layer %d is'%(row,col,layers[i]))
                # print(boxes[row*layers[i]+col])
                #print('no %d box'%(row*layers[i]+col))
    # print("shape of default boxes")
    # print(boxes)
    # print("default boxes are")
    # print(boxes)
    boxes = boxes.reshape((number_of_boxes*4,8))
    boxes[boxes<0] = 0
    # for i in range(0,boxes.shape[0]):
    #     for j in range(0,boxes.shape[1]):
    #         if boxes[i][j] < 0 :
    #             boxes[i][j] = 0
    # print("default box shape is ")
    # print(boxes.shape)
    # print("default box ")
    # print(boxes)
    #print("box shape")
    #print(boxes.shape)
    return boxes


#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)



def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    #compute iou between the default bounding boxes and the ground truth bounding box

    ious = iou(boxs_default, x_min,y_min,x_max,y_max)
    
    #ious_true = ious>threshold
    #TODO:
    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    centre_x = format((x_min + x_max) / 2, '.2f')
    centre_y = format((y_min + y_max) / 2, '.2f')
    w = x_max - x_min
    h = y_max - y_min
    #print("in match")
    for i in range(0,ious.size):
        if ious[i]> threshold:
            # print('default box in match %d is ' %(i))
            # print(boxs_default[i])
            tx = (float(centre_x)-float(boxs_default[i][0]))/float(boxs_default[i][2])
            ty = (float(centre_y)-float(boxs_default[i][1]))/float(boxs_default[i][3])
            tw = np.log(w/boxs_default[i][2])
            th = np.log(h/boxs_default[i][3])
            ann_box[i] = [tx,ty,tw,th]
            ann_confidence[i,cat_id] = 1
            ann_confidence[i,-1] = 0
            # print('ann_box  %d  in match is ' %(i))
            # print(ann_box[i])
            # print('ann_confidence in match is ')
            # print(ann_confidence[i])


    ious_true = np.argmax(ious)
    #TODO:
    #make sure at least one default bounding box is used
    #update ann_box and ann_confidence (do the same thing as above)
    tx = (float(centre_x) -float( boxs_default[ious_true][0])) /float(boxs_default[ious_true][2])
    ty = (float(centre_y) - float(boxs_default[ious_true][1])) / float(boxs_default[ious_true][3])
    tw = np.log(w / boxs_default[ious_true][2])
    th = np.log(h / boxs_default[ious_true][3])
    ann_box[ious_true] = [tx, ty, tw, th]

    ann_confidence[ious_true, cat_id] = 1
    ann_confidence[ious_true, -1] = 0
    return ann_box, ann_confidence

def random_crop(image,w_crop,h_crop):


    x = random.randint(0, image.shape[1] - w_crop)
    y = random.randint(0, image.shape[0] - h_crop)
    print("random crop x is " + str(x))
    print("random crop y is "+ str(y))

    image = image[y:y+h_crop,x:x+w_crop]
    # print("image shape is ")
    # print(image.shape)
    return x,y,image



class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size
        image_len = len(self.img_names)
        self.train_size = int(image_len*0.9)
        self.test_size = image_len-self.train_size
        self.train_name_list, self.test_name_list = data.random_split(self.img_names, [self.train_size, self.test_size])
        #notice:
        #you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train

    def __len__(self):
        if self.train:
            return  self.train_size
        else:
            return self.test_size


    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #print(self.box_num)
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        if self.train:
            img_name = self.imgdir + self.train_name_list[index]

        else:
            img_name = self.imgdir + self.test_name_list[index]

        ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        
        #TODO:
        #1. prepare the image [3,320,320], by reading image "img_name" first.
        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        image = cv2.imread(img_name)
        # print("original image size is ")
        # print(image.shape)
        txt_file = open(ann_name)
        lines = txt_file.readlines()
        h_ratio = 1
        w_ratio = 1
        h_resized = False
        w_resized = False
        img_height,img_width,_ = image.shape
        # if image.shape[0]!=self.image_size:
        #     h_ratio = round(self.image_size/image.shape[0],2)
        #     image = cv2.resize(image,(image.shape[1],self.image_size))
        #     h_resized = True
            # print("h_ration is ")
            # print(h_ratio)
        # if image.shape[1] != self.image_size:
        #     w_ratio = round(self.image_size/image.shape[1],2)
        #     image = cv2.resize(image,(self.image_size,image.shape[0]))
        #     w_resized = True
            # print("w_ratio is ")
            # print(w_ratio)
        image = np.array(image)


        #x,y,image = random_crop(image,self.image_size,self.image_size)
        x = 0
        y = 0
        # print("image shape after crop")
        # print(image.shape)
        for line in lines:
            # print("ann box is ")
            # print(line)
            class_id = int(line.split(' ')[0])
            x_min = float(line.split(' ')[1])
            y_min = float(line.split(' ')[2])
            box_w = float(line.split(' ')[3])
            box_h = float(line.split(' ')[4])
            x_max = x_min+box_w
            y_max = y_min+box_h
            x_min_norm = x_min/img_width
            y_min_norm = y_min/img_height
            x_max_norm = x_max/img_width
            y_max_norm = y_max/img_height
            # if h_resized:
            #     y_min *= h_ratio
            #     y_max *= h_ratio
            #     h *= h_ratio
            # if w_resized:
            #     x_min *= w_ratio
            #     x_max *= w_ratio
            #     w *= w_ratio
            # x_min = round(x_min,2)
            # y_min = round(y_min,2)
            # x_max = round(x_max,2)
            # x_min = round(x_min,2)
            # x_min = x_min - x
            # y_min = y_min - y
            # x_max = x_min + w
            # y_max = y_min + h
            # x_min = max(0, x_min)
            # y_min = max(0, y_min)
            # x_max = min(self.image_size, x_max)
            # y_max = min(self.image_size, y_max)
            # print("after crop before normalize")
            # print((x_min,y_min,x_max,y_max,w,h))
            # x_min = round(x_min/self.image_size,2)
            # y_min = round(y_min/self.image_size,2)
            # x_max = round(x_max/self.image_size,2)
            # y_max = round(y_max/self.image_size,2)
            # w = round(x_max - x_min,2)
            # h = round(y_max - y_min,2)
            # print("after crop the ann_box is ")
            # print((x_min,y_min,x_max,y_max,w,h))
            ann_box, ann_confidence = match(ann_box, ann_confidence, self.boxs_default, self.threshold, class_id, x_min_norm,
                                            y_min_norm, x_max_norm, y_max_norm)


        # class_id = line[0].split(' ')[0]
        # x_min = float(line[0].split(' ')[1])/image.shape[1]
        # y_min = float(line[0].split(' ')[2])/image.shape[0]
        # w = float(line[0].split(' ')[3])
        # h = float(line[0].split(' ')[4])
        # x_max = (x_min+w)/image.shape[1]
        # y_max = (y_min+h)/image.shape[0]
        image = cv2.resize(image, (320, 320))
        # image = np.array(image)

        #to use function "match":
        #ann_box,ann_confidence = match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        
        #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        image = np.transpose(image,(2,0,1))
        # print(ann_box.shape)
        # print(ann_confidence.shape)
        # print("return image size ")
        # print(image.shape)
        #image [320,320,3]

        return image, ann_box, ann_confidence
