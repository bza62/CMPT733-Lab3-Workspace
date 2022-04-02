import argparse
import os
import numpy as np
import time
import cv2
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib.pyplot as plt
from dataset import *
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
#please google how to use argparse
#a short intro:
#to train: python main.py # if width > 320 and height > 320:

#to result:  python main.py --result


class_num = 4 #cat dog person background

num_epochs = 30
batch_size = 10


boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])


#Create network
network = SSD(class_num)
network.cuda()
cudnn.benchmark = True

pred_confidence_all, pred_box_all, ann_confidence_all, ann_box_all, images_all, ann_name_all = [], [], [], [], [], []

if not args.test:
    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = True, image_size=320)
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    train_losses = []
    val_losses = []
    optimizer = optim.Adam(network.parameters(), lr = 1e-4)
    #feel free to try other optimizers and parameters.
    start_time = time.time()
    use_chekcpoint = False
    for epoch in range(num_epochs):
        #TRAINING
        #
        if epoch==0:
            network.load_state_dict(torch.load('checkpoint/32_1network.pth'))

        if use_chekcpoint:
            network.load_state_dict(torch.load(
                 'checkpoint/'+str(epoch-1)+'network.pth'))
        use_chekcpoint = False
        network.train()

        avg_loss = 0
        avg_count = 0
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_,ann_name_ = data

            # print("i is "+str(i))
            # print("ann_name_ length is "+ str(len(ann_name_)))
            # print("image length is "+str(len(images_)))

            # print(ann_box_.shape)
            # print(ann_confidence_.shape)
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            #print("pre_confidence length is "+str(len(pred_confidence)))
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            #result visualize_pred
            ann_confidence_all.append(ann_confidence)
            ann_box_all.append(ann_box)
            pred_confidence_all.append(pred_confidence)
            pred_box_all.append(pred_box)
            images_all.append(images_)
            ann_name_all.append(ann_name_)
            #pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
            #pred_box_ = pred_box[0].detach().cpu().numpy()
            loss_net.backward()
            optimizer.step()
            avg_loss += loss_net.data
            avg_count += 1
            pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
            pred_box_ = pred_box[0].detach().cpu().numpy()
            print("batch %d loss is %d"%(i,avg_loss) )
            # print("name is "+str(ann_name_))
            # for batch_i in len(images_):
            #     pred_confidence_ = pred_confidence[batch_i].detach().cpu().numpy()
            #     pred_box_ = pred_box[batch_i].detach().cpu().numpy()
            #     visualize_pred(epoch, "train", pred_confidence_, pred_box_, ann_confidence_[batch_i].numpy(),
            #                    ann_box_[batch_i].numpy(),images_[batch_i].numpy(), boxs_default)
            #


        avg_loss = avg_loss
        print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))
        train_losses.append(avg_loss/avg_count)

        # save checkpoint
        if epoch % 10 ==9:
            # save last network
            use_chekcpoint = True
            print('saving net...')
            torch.save(network.state_dict(),
                       'checkpoint/' + str(
                           epoch) + 'network.pth')
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
#        visualize_pred(epoch, "train",' ', pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(),images_[0].numpy(), boxs_default)
        #pred_confidence_all, pred_box_all, ann_confidence_all, ann_box_all, images_all, ann_name_all
        #visualize
    #     if epoch == num_epochs-1:
    #         total_batch = len(ann_confidence_all)
    #         # print("ann_name")
    #         # print(len(ann_confidence_all))
    #         # print(len(ann_confidence_all[-1]))
    # #        batch_no = len()
    #         for i in range(0,total_batch):
    #             batch_i = len(ann_confidence_all[i])
    #             for j in range(0,batch_i):
    #                 #print("ann_name")
    #                 #print(ann_name_all[i][j])
    #
    #                 pred_confidence_batch = pred_confidence_all[i][j].detach().cpu().numpy()
    #                 pred_box_batch = pred_box_all[i][j].detach().cpu().numpy()
    #
    #                 # print("pred_confidence")
    #                 # print(pred_confidence_all[i][j].shape)
    #                 # print("pred_box")
    #                 # print(pred_box_all[i][j])
    #                 # print("pred_confidence")
    #                 # print(pred_confidence_all[i][j])
    #                 # print("pred_box")
    #                 #print(pred_box_all[i][j].shape)
    #
    #                 visualize_pred(i*batch_size+j,"train",ann_name_all[i][j],pred_confidence_batch,pred_box_batch,
    #                                ann_confidence_all[i][j].detach().cpu().numpy(), ann_box_all[i][j].detach().cpu().numpy(),
    #                                images_all[i][j].detach().cpu().numpy(),boxs_default)
        #visualize_pred(epoch,"train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        #draw_after_nms(epoch,"train", pred_confidence_, pred_box_,  images_[3].numpy(), boxs_default)


        #VALIDATION
        network.eval()
        
        # TODO: split the dataset into 90% training and 10% validation
        # use the training set to train and the validation set to evaluate
        val_avg_loss = 0
        val_avg_count = 0
        # for i, data in enumerate(dataloader_test, 0):
        #     print("in validation")
        #     images_, ann_box_, ann_confidence_,ann_name_ = data
        #     images = images_.cuda()
        #     ann_box = ann_box_.cuda()
        #     ann_confidence = ann_confidence_.cuda()
        #
        #     pred_confidence, pred_box = network(images)
        #     loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
        #
        #     val_avg_loss += loss_net
        #     val_avg_count += 1
        #     pred_confidence_ = pred_confidence.detach().cpu().numpy()
        #     pred_box_ = pred_box.detach().cpu().numpy()
        #
        #     #optional: implement a function to accumulate precision and recall to compute mAP or F1.
        #     #update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_,thres)
        #
        # #visualize
        # # print("ann_name_all in validation is ")
        # # print(ann_name_all)
        # val_losses.append(val_avg_loss / val_avg_count)
        # pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        # pred_box_ = pred_box[0].detach().cpu().numpy()
        # visualize_pred(epoch,"val", ' ',pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(),
        #                images_[0].numpy(), boxs_default)

        # for j in range(batch_size):
            #     pred_confidence_ = pred_confidence[j].detach().cpu().numpy()
            #     pred_box_ = pred_box[j].detach().cpu().numpy()
            #     visualize_pred(i+j,"val", ann_name_[j],pred_confidence_, pred_box_,
            #                 ann_confidence_[j].detach().cpu().numpy(), ann_box_[j].detach().cpu().numpy(), images_[j].detach().numpy(), boxs_default)

        #optional: compute F1
        #F1score = 2*precision*recall/np.maximum(precision+recall,1e-8)
        #print(F1score)
        
        #save weights

    # x = np.arange(num_epochs)
    # print(x)
    # fig, ax = plt.subplots()
    # ax.plot(x, train_losses.detach().cpu().numpy(), 'go-', label='train loss')
    # ax.plot(x,val_losses.detach().cpu().numpy(), 'ro-', label='val loss')
    # ax.legend()
    # plt.show()
    # plt.plot()

else:
    #TEST
    dataset_test = COCO("data/test/images/", "data/test/annotations/", class_num, boxs_default, train = False, image_size=320)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('checkpoint/9network.pth'))
    network.eval()
    
    for i, data in enumerate(dataloader_test, 0):

        images_, ann_box_, ann_confidence_,ann_name_ = data

        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)
        # print(len(pred_confidence_));raise Exception
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        
        #pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        
        #TODO: save predicted bounding boxes and classes to a txt file.
        #you will need to submit those files for grading this assignment
        # ann_confidence_all.append(ann_confidence)
        # ann_box_all.append(ann_box)
        # pred_confidence_all.append(pred_confidence)
        # pred_box_all.append(pred_box)
        # images_all.append(images_)
        # visualize_pred(i,"result", ' ' ,pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(),
        #                images_[0].numpy(), boxs_default)
        cv2.waitKey(1000)
        ann_name_all.append(ann_name_)
        batch_size = len(pred_box)


        name = i
        name = str(name).zfill(5)

        # print(pred_confidence_.shape)
        # print(images_.shape)
        # print(pred_confidence[0].shape);raise Exception
        visualize_pred(i , "test", name, pred_confidence[0].detach().cpu().numpy(), pred_box[0].detach().cpu().numpy(),ann_confidence[0].detach().cpu().numpy(), ann_box[0].detach().cpu().numpy(),images[0].detach().cpu().numpy(), boxs_default)

        # for i in range(0, total_batch):
        #     for j in range(0,batch_size):
        #         name = i*batch_size+j
        #         name = str(name).zfill(5)
        #         print(name)
        #         pred_confidence_ = pred_confidence_all[i][j].detach().cpu().numpy()
        #         pred_box_ = pred_box_all[i][j].detach().cpu().numpy()
        #         visualize_pred(i*batch_size+j,"result",name,pred_confidence_,pred_box_,
        #                        ann_confidence_all[i][j].detach().cpu().numpy(), ann_box_all[i][j].detach().cpu().numpy(),images_all[i][j].detach().cpu().numpy(),boxs_default)

        # visualize_pred(i,"result", ' ' ,pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        #cv2.waitKey(1000)



