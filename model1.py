import os
import random
import numpy as np

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




def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: write a loss function for SSD
    #
    #For confidence (class labels), use cross entropy (F.cross_entropy)
    #You can try F.binary_cross_entropy and see which loss is better
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.
    # print("pre_box shape "+ str(pred_box.shape))
    # print(pred_box.shape[0]*pred_box.shape[1])
    # print(pred_box.shape[2])
    batch_size = pred_box.shape[0]
    pred_box = pred_box.reshape((pred_box.shape[0]*pred_box.shape[1],pred_box.shape[2]))
    #ann_box = ann_box.cpu().detach().numpy()
    # print("ann_box")
    # print(ann_box.shape)
    ann_box = ann_box.reshape((ann_box.shape[0]*ann_box.shape[1],ann_box.shape[2]))
    pred_confidence = pred_confidence.reshape((pred_confidence.shape[0]*pred_confidence.shape[1],pred_confidence.shape[2]))
    #ann_confidence = ann_confidence.cpu().detach().numpy()
    # print("ann_confidence")
    # print(ann_confidence.shape)
    ann_confidence =ann_confidence.reshape((ann_confidence.shape[0]*ann_confidence.shape[1],ann_confidence.shape[2]))
    t_noobj = ann_confidence[:,3]
    t_obj = 1-t_noobj
    noobj = t_noobj.cpu().numpy()
    obj = t_obj.cpu().numpy()
    t_noobj = np.where(noobj==1)
    t_obj = np.where(obj==1)
    pred_confidence_obj = pred_confidence[t_obj]
    pred_confidence_noobj = pred_confidence[t_noobj]

    ann_confidence_obj = ann_confidence[t_obj]
    ann_confidence_noobj = ann_confidence[t_noobj]

    pred_box_obj = pred_box[t_obj]
    ann_box_obj = ann_box[t_obj]
    L_cls = F.cross_entropy(pred_confidence_obj,ann_confidence_obj)+\
            3*F.cross_entropy(pred_confidence_noobj,ann_confidence_noobj)
    L_box = F.smooth_l1_loss(pred_box_obj,ann_box_obj)
    #print("loss is "+str(L_box+L_cls))
    return L_box+L_cls








class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        
        #TODO: define layers
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1,stride=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1,stride=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1,stride=2),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1,stride=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1,stride=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1,stride=2),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.conv8 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1,stride=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1,stride=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.conv10 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1,stride=2),nn.BatchNorm2d(512),nn.ReLU(inplace=True))
        self.conv11 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1,stride=1),nn.BatchNorm2d(512),nn.ReLU(inplace=True))
        self.conv12 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1,stride=1),nn.BatchNorm2d(512),nn.ReLU(inplace=True))
        self.conv13 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1,stride=2),nn.BatchNorm2d(256),nn.ReLU(inplace=True))

        self.conv14 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.conv15 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1,stride=2),nn.BatchNorm2d(256),nn.ReLU(inplace=True))

        self.conv16 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv17 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))


        self.conv18 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv19 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride = 3,padding=1),nn.ReLU(inplace=True))
        #layer4
        self.layer = nn.Sequential( nn.Conv2d(256, 16, kernel_size=3,padding=1,stride=1),nn.BatchNorm2d(16),nn.ReLU(True))

        self.layer4 =nn.Sequential( nn.Conv2d(256, 16, kernel_size=1,stride=1),nn.ReLU(True))
        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        
        #TODO: define forward
        
        #should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        x = self.conv1(x)
        #print(" after conv1 shape "+str(x.shape))
        x = self.conv2(x)
        #print("after conv2 shape "+str(x.shape))
        x = self.conv3(x)
        #print("after conv3 shape "+str(x.shape))
        x = self.conv4(x)
        #print("after conv4 shape " + str(x.shape))
        x = self.conv5(x)
        #print("after conv5 shape " + str(x.shape))
        x = self.conv6(x)
        #print("after conv6 shape " + str(x.shape))
        x = self.conv7(x)
        #print("after conv7 shape " + str(x.shape))
        x = self.conv8(x)
        #print("after conv8 shape " + str(x.shape))
        x = self.conv9(x)
        #print("after conv9 shape " + str(x.shape))
        x = self.conv10(x)
        #print("after conv10 shape " + str(x.shape))
        x = self.conv11(x)
        #print("after conv11 shape " + str(x.shape))
        x = self.conv12(x)
        #print("after conv12 shape " + str(x.shape))
        x = self.conv13(x)
        #print("after conv13 shape " + str(x.shape))
        layer1_1 = self.layer(x)
        layer1_2 = self.layer(x)
        #print("layer1 "+str(layer1_1.shape))
        x = self.conv14(x)
        #print("after 14 "+str(x.shape))
        x = self.conv15(x)
        #print("after 15 " + str(x.shape))
        layer2_1 = self.layer(x)
        layer2_2 = self.layer(x)
        #print("layer 2 "+str(layer2_1.shape))
        x = self.conv16(x)
        #print("after 16 " + str(x.shape))
        x = self.conv17(x)
        #print("after 17 " + str(x.shape))
        layer3_1 = self.layer(x)
        layer3_2 = self.layer(x)
        #print("layer3 "+str(layer3_1.shape))
        x = self.conv18(x)
        #print("after 18 " + str(x.shape))
        x = self.conv19(x)
        #print("after 19 " + str(x.shape))
        layer4_1 = self.layer4(x)
        layer4_2 = self.layer4(x)
        #print("layer 4 "+str(layer4_1.shape))
        batch_size = layer1_1.shape[0]
        #10*10
        #layer1_1 = layer1_1.cpu().detach().numpy()
        layer1_1 = layer1_1.reshape((batch_size,16,100))
        #layer1_2 = layer1_2.detach().cpu().numpy()
        layer1_2 = layer1_2.reshape((batch_size,16,100))
        #5*5
        #layer2_1 = layer2_1.cpu().detach().numpy()
        layer2_1 = layer2_1.reshape((batch_size,16,25))
        #layer2_2 = layer2_2.cpu().detach().numpy()
        layer2_2 = layer2_2.reshape((batch_size,16,25))
        #3*3
        #layer3_1 = layer3_1.cpu().detach().numpy()
        layer3_1 = layer3_1.reshape((batch_size,16,9))
        #layer3_2 = layer3_2.cpu().detach().numpy()
        layer3_2 = layer3_2.reshape((batch_size,16,9))
        #1*1
        #layer4_1 = layer4_1.cpu().detach().numpy()
        layer4_1 = layer4_1.reshape((batch_size,16,1))
       # layer4_2 = layer4_2.cpu().detach().numpy()
        layer4_2 = layer4_2.reshape((batch_size,16,1))

        bboxes = torch.cat((layer1_1,layer2_1,layer3_1,layer4_1),axis=2)
        confidence = torch.cat((layer1_2,layer2_2,layer3_2,layer4_2),axis=2)
        bboxes = torch.permute(bboxes,(0,2,1))
        bboxes = bboxes.reshape((batch_size,540,4))
        confidence = torch.permute(confidence,(0,2,1))
        confidence = confidence.reshape((batch_size,540,4))
        #print("confidence shape " + str(confidence.shape))
        #confidence = torch.from_numpy(confidence)
        #confidence = F.softmax(confidence, dim = 2)
        #confidence = confidence.numpy()
        #print("confidence shape "+str(confidence.shape))
        #print("bboxes shape "+str(bboxes.shape))
        return confidence,bboxes










