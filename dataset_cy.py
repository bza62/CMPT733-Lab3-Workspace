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
from PIL import Image
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
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    boxes = []
    box_num = 4 * (10 * 10 + 5 * 5 + 3 * 3 + 1 * 1)
    for k in range(len(layers)):
        start_x = (1/layers[k])/2
        start_y = (1/layers[k])/2
        grid_side = 1/layers[k]
        ssize = small_scale[k]
        lsize = large_scale[k]
        for i in range(layers[k]*layers[k]):
            box_for_4 = []
            row = i//layers[k]
            col = i % layers[k]
            x_center = start_x+row*grid_side
            y_center = start_y+col*grid_side
            box_for_4.append([x_center,y_center,ssize,ssize,x_center-ssize/2,y_center-ssize/2,x_center+ssize/2,y_center+ssize/2])
            box_for_4.append([x_center,y_center,lsize,lsize,x_center-lsize/2,y_center-lsize/2,x_center+lsize/2,y_center+lsize/2])
            box_for_4.append([x_center,y_center,lsize*np.sqrt(2),lsize/np.sqrt(2),x_center-(lsize*np.sqrt(2))/2,y_center-(lsize/np.sqrt(2))/2,x_center+(lsize*np.sqrt(2))/2,y_center+(lsize/np.sqrt(2))/2])
            box_for_4.append([x_center,y_center,lsize/np.sqrt(2),lsize*np.sqrt(2),x_center-(lsize/np.sqrt(2))/2,y_center-(lsize*np.sqrt(2))/2,x_center+(lsize/np.sqrt(2))/2,y_center+(lsize*np.sqrt(2))/2])
            boxes.append(box_for_4)
    boxes = np.array(boxes) 
    boxes = np.reshape(boxes,(box_num,8))         
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
    #Calculate the coverage between two bounding boxes
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    res = inter/np.maximum(union,1e-8)
    # ious_true = res>0.5
    # print(ious_true)
    #The closer the return value is to 1, the more consistent the two bounding boxes are
    return res



def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box

    # for i in range(boxs_default.shape[0]):

    #compute iou between the default bounding boxes and the ground truth bounding box
    x_center, y_center, box_width, box_height = (x_min+x_max)/2, (y_min+y_max)/2, (x_max-x_min), (y_max-y_min)
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)
    ious_true = ious>threshold
    true_idx = [index for (index,value) in enumerate(ious_true) if value == True]
    # print(true_idx)
    for i in true_idx:
        ann_confidence[i][3] = 0
        ann_confidence[i][cat_id] = 1
        ann_box[i][0] = (x_center-boxs_default[i][0])/boxs_default[i][2]
        ann_box[i][1] = (y_center-boxs_default[i][1])/boxs_default[i][3]
        ann_box[i][2] = np.log(box_width/boxs_default[i][2])
        ann_box[i][3] = np.log(box_height/boxs_default[i][3])
    #TODO:
    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    
    ious_true = np.argmax(ious)
    # print(ious_true)
    ann_confidence[ious_true][3] = 0
    ann_confidence[ious_true][cat_id] = 1
    ann_box[ious_true][0] = (x_center-boxs_default[ious_true][0])/boxs_default[ious_true][2]
    ann_box[ious_true][1] = (y_center-boxs_default[ious_true][1])/boxs_default[ious_true][3]
    ann_box[ious_true][2] = np.log(box_width/boxs_default[ious_true][2])
    ann_box[ious_true][3] = np.log(box_height/boxs_default[ious_true][3])
    #TODO:
    #make sure at least one default bounding box is used
    #update ann_box and ann_confidence (do the same thing as above)



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
        
        #notice:
        #you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train

    def __len__(self):
        image_train_number = int(len(self.img_names)*0.9)
        if self.train:
            return len(self.img_names[0:image_train_number])
        else:
            return len(self.img_names[image_train_number:len(self.img_names)])

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        # print(os.listdir(self.imgdir))
        # data spliting
        image_train_number = int(len(self.img_names)*0.9)
        
        image_train = self.img_names[0:image_train_number]
        image_test = self.img_names[image_train_number:len(self.img_names)]
        if self.train:       
            img_name = self.imgdir+image_train[index]
            ann_name = self.anndir+image_train[index][:-3]+"txt"
        else:
            img_name = self.imgdir+image_test[index]
            ann_name = self.anndir+image_test[index][:-3]+"txt"
        
        #TODO:
        #1. prepare the image [3,320,320], by reading image "img_name" first.
        image = Image.open(img_name)
        
        # image =transforms.Resize([self.image_size,self.image_size])(image)
        image = np.asarray(image)
        if len(image.shape) == 2:
            img = np.zeros([image.shape[0], image.shape[1], 3])
            img[:,:,0] = image
            img[:,:,1] = image
            img[:,:,2] = image
            image = img
        height = image.shape[0]
        # width = 640,height = 480
        width = image.shape[1]
        # print(width)
        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        with open(ann_name, 'r') as f:
            for line in f:
                class_id, x_min, y_min, box_width, box_height = line.split()
                class_id = int(class_id)
                x_min = float(x_min)
                y_min = float(y_min)
                box_width = float(box_width)
                box_height = float(box_height)
                x_min_norm = x_min/width
                y_min_norm = y_min/height
                box_width_norm = box_width/width
                box_height_norm = box_height/height
                x_max_norm,y_max_norm = x_min_norm+box_width_norm, y_min_norm+box_height_norm
                match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min_norm,y_min_norm,x_max_norm,y_max_norm)
        
        #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        
        #to use function "match":
        #match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        
        #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        image = torch.from_numpy(image)
        image = image.type(torch.FloatTensor)
        image =  torch.permute(image, (2, 0, 1))
        # print(image.shape)
        image =transforms.Resize([self.image_size,self.image_size])(image)
        ann_box = torch.from_numpy(ann_box)
        ann_confidence = torch.from_numpy(ann_confidence)
        return image, ann_box, ann_confidence



###############################################################################
########################test code##############################################
###############################################################################
###############################################################################
# class_num = 4
# boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])
# # print(boxs_default)
# # # res = iou(boxs_default,0, 0, 0.1, 0.1)
# image, ann_box, ann_confidence = COCO("CMPT733-Lab3-Workspace/data/data/data/train/images/", "CMPT733-Lab3-Workspace/data/data/data/train/annotations/", class_num, boxs_default, train = True, image_size=320).__getitem__(9)

# from model import *
# image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
# ann_box = ann_box.reshape((1,ann_box.shape[0],ann_box.shape[1]))
# ann_confidence = ann_confidence.reshape((1,ann_confidence.shape[0],ann_confidence.shape[1]))
# network = SSD(class_num)
# pred_conf,pred_bbox = network.forward(image)
# # print(sum(ann_confidence))
# # np.set_printoptions(threshold=np.inf)
# # print(ann_confidence)
# x = SSD_loss(pred_conf, pred_bbox, ann_confidence, ann_box)
# print(x)