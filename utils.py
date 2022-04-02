import numpy as np
import cv2
from dataset import iou
import  re
import torch.nn as nn
from scipy.special import softmax
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes
#cat dog person
def shape_of_box(offset, box_default):
# return start point and end point

    centre_x = offset[0]*box_default[2]+box_default[0]
    centre_y = offset[1]*box_default[3]+box_default[1]
    w = np.exp(offset[2])*box_default[2]
    h = np.exp(offset[3])*box_default[3]

    start_point,end_point = start_end_point(centre_x,centre_y,w,h)
    return start_point,end_point

def start_end_point(centre_x,centre_y,w,h):

    start_point = (max(int((centre_x - w / 2)*320),10), max(int((centre_y - h / 2)*320),10))
    end_point = (min(int((centre_x + w / 2)*320),300), min(int((centre_y + h / 2)*320),300))

    return start_point,end_point


def visualize_pred(epoch,windowname,ann_name ,pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    #input:
    #windowname   # visualize_pred("train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(),
            #                images_[0].numpy(), boxs_default)    -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    #print(image_)
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    # print("image shape ")
    # print(image.shape)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image5 = np.zeros(image.shape, np.uint8)
    image6 = np.zeros(image.shape, np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    image5[:] = image[:]
    image6[:] = image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)


    #ann_confidence = softmax(ann_confidence, axis=1)

    pred_confidence = softmax(pred_confidence,axis=1)

    #print("gt confidence is " + ann_confidence);raise Exception
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):

            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO:
                #image1: draw ground truth bounding boxes on image1
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)

                #you can use cv2.rectangle as follows:
                #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                #thickness = 2

                start_point_gt,end_point_gt = shape_of_box(ann_box[i],boxs_default[i])

                start_point_default,end_point_default = start_end_point(boxs_default[i][0],boxs_default[i][1],boxs_default[i][2],boxs_default[i][3])
                # print("the gt is ")
                # print(start_point_gt)
                # print(end_point_gt)
                # print(end_point_default)
                # print(start_point_default)

                cv2.rectangle(image1,start_point_gt,end_point_gt,color=colors[j],thickness=2)
                # cv2.imshow(windowname + " [[gt_box,gt_dft],[pd_box,pd_dft]]", image1)
                # cv2.waitKey(0)
                cv2.setNumThreads(0)
                cv2.ocl.setUseOpenCL(False)
                cv2.rectangle(image2,start_point_default,end_point_default,color=colors[j],thickness=2)
                cv2.setNumThreads(0)
                cv2.ocl.setUseOpenCL(False)
                # cv2.imshow(windowname + " [[gt_box,gt_dft],[pd_box,pd_dft]]", image2)
                # cv2.waitKey(0)
    confidence_, box_, boxs_default_,index= [],[],[],[]
    color_index = []
    #pred


    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.8:
                #TODO:
                #image3: draw network-predicted bounding boxes on image3
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                confidence_.append(pred_confidence[i,j])
                box_.append(pred_box[i])
                boxs_default_.append(boxs_default[i])
                index.append(i)
                color_index.append(j)
                # print("The class is %d "%(j))
                # print('box %d has a confidence of %f'%(i,pred_confidence[i,j]))
                # print("offset is ")
                # print(pred_box[i])

                start_point_pred, end_point_pred = shape_of_box(pred_box[i], boxs_default[i])

                start_point_default, end_point_default = start_end_point(boxs_default[i][0], boxs_default[i][1],boxs_default[i][2], boxs_default[i][3])
                # print("start point and end poit are")
                # print(start_point_pred)
                # print(end_point_pred)
                #box_.append(start_point_pred)
                #box_.append(end_point_pred)

                cv2.rectangle(image3, start_point_pred, end_point_pred, color=colors[j], thickness=2)
                cv2.setNumThreads(0)
                cv2.ocl.setUseOpenCL(False)
                cv2.rectangle(image4, start_point_default, end_point_default, color=colors[j], thickness=2)
                cv2.setNumThreads(0)
                cv2.ocl.setUseOpenCL(False)
    #use_nms = False
    # print("confidence before nms is ")
    # print(confidence_)
    # print(box_)

    #if len(box_) >1:
    use_nms = True
    # print("before nms confidence is ")
    # print(confidence_)

    results = non_maximum_suppression(confidence_,box_,boxs_default_,0.5,0.5)

    if windowname != "val":
        print(ann_name)
        ann_name = ann_name[17:22]
        print(ann_name)
       # print(ann_name)
        f = open('data/'+windowname+'/pred_annotations/'+str(ann_name)+'.txt','w')

    for i in results:
       # print("index i is "+str(index[i]))
        #print("result in utils is ")

        start_point_pred, end_point_pred = shape_of_box(pred_box[index[i]], boxs_default[index[i]])
        start_point_default, end_point_default = start_end_point(boxs_default[index[i]][0], boxs_default[index[i]][1],
                                                                 boxs_default[index[i]][2], boxs_default[index[i]][3])
        if windowname != "val":
            w = str(end_point_pred[0]-start_point_pred[0])
            h = str(end_point_pred[1]-start_point_pred[1])
            x = str(start_point_pred[0])
            y = str(start_point_pred[1])
            #print(str(color_index[i])+' '+x+' ' +y+' '+w+' '+h)
            f.write(str(color_index[i])+' '+x+' ' +y+' '+w+' '+h+'\n')
        #color = cat[results[i]]
        # print("start and end point after nms")
        # print(start_point_pred)
        # print(end_point_pred)
        #
        # print("color is "+ str(color_index[i]))
        cv2.rectangle(image5, start_point_pred, end_point_pred, color= colors[color_index[i]], thickness=2)
        cv2.rectangle(image6, start_point_default, end_point_default, color=colors[color_index[i]], thickness=2)
        # cv2.setNumThreads(0)
        # cv2.ocl.setUseOpenCL(False)
    #combine four images into one

    h,w,_ = image1.shape
    image = np.zeros([h*3,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:2*h,:w] = image3
    image[h:2*h,w:] = image4
    if use_nms:
        image[2*h:,:w] = image5
        image[2*h:,w:] = image6
    # cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    # cv2.waitKey(0)
    #print("epoch is "+ str(epoch))
    cv2.imwrite('data/'+windowname+'/'+'result/'+ann_name+'.jpg',image)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.


def iou(start_point_max,end_point_max,start_point_i,end_point_i):
    # input:
    # boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    # x_min,y_min,x_max,y_max -- another box (box_r)
    # print("max points")
    # print(start_point_max)
    # print(end_point_max)
    # print("point i")
    # print(start_point_i)
    # print(end_point_i)
    # output:
    # ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    inter = (np.minimum(end_point_i[0],end_point_max[0])-np.maximum(start_point_max[0],start_point_i[0])) * \
            (np.minimum(end_point_i[1],end_point_max[1])-np.maximum(start_point_max[1],start_point_i[1]))
    # print("inter w is "+str(np.minimum(end_point_i[0],end_point_max[0])-np.maximum(start_point_max[0],start_point_i[0])))
    # print("inter h is "+str(np.minimum(end_point_i[1],end_point_max[1])-np.maximum(start_point_max[1],start_point_i[1])))
    # print("area of inter is "+str(inter))

    if inter<0:
        inter = 0
        # print("after 0 inter is ")
        # print(inter)
    area_a = (end_point_i[0] - start_point_i[0]) * (end_point_i[1] - start_point_i[1])
    area_b = (end_point_max[0] - start_point_max[0]) * (end_point_max[1] - start_point_max[1])
    # print("area a is "+str(area_a))
    # print("area b is "+str(area_b))
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-8)




def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.1, threshold=0.5):
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    size = len(box_)
    results = []

    print(len(confidence_))
    b = np.array([0,0,0,0])
    # print("confidence is ")
    # print((confidence_))
    confidence_ = np.array(confidence_)
    # print("confidence is ")
    # print(confidence_)
    for i in range(0,size):
        # print("before confidence is ")
        # print(confidence_)
        if confidence_.max() == 0:

            break
        max_index = np.argmax(confidence_)
        # print("max_index")
        # print(max_index)
        #max_box = box_[max_index]
        results.append(max_index)
        print("max_index before is %d "%(max_index))
        for j in range(0,size):
            #box_[j] = np.array(box_[j])
            if j!= max_index:
                if  confidence_[j]!=0 :
                    print("compare %d box"%(j))

                    # print("box_ in nms")
                    # print(box_[max_index])
                    # print(boxs_default[max_index])
                    # print(box_[max_index][0:3])
                    # print(boxs_default[max_index][0:3])
                    #print("box index is "+)
                    start_point_max, end_point_max = shape_of_box(box_[max_index], boxs_default[max_index])
                    start_point_i,end_point_i = shape_of_box(box_[j],boxs_default[j])
                    inter = iou(start_point_max,end_point_max,start_point_i,end_point_i)
                    #print("inter is "+str(inter))
                    # print("max start point and end point are ")
                    # print(start_point_max)
                    # print(end_point_max)
                    # print("i start point and end point are ")
                    # print(start_point_i)
                    # print(end_point_i)
                    if inter>overlap:
                        print("delete index is "+ str(j))
                        box_[j] = np.array([0,0,0,0])
                        boxs_default[j] = np.array([0,0,0,0])
                        confidence_[j] = 0
        box_[max_index] = np.array([0,0,0,0])
        confidence_[max_index] = 0

        # print("confidence is ")
        # print(confidence_)
        # confidence_[max_index] = 0
    # print("result in nms is ")
    print(len(results))
    #print(results)
    return results
    # box_result = []
    # cat = confidence_[:,0]
    #
    # dog = confidence_[:,1]
    # person = confidence_[:,2]
    # classes_confidence = np.array([cat,dog,person])
    # for class_i in range(0,classes_confidence.shape[0]):
    #     max_index = np.argmax(classes_confidence[class_i]) #index of box that has highest confidence
    #     max_value =classes_confidence[class_i][max_index]
    #     if max_value>threshold:
    #         for box_i in range(0, box_.shape[0]):
    #
    #             if box_i != max_index:
    #                 start_point_max,end_point_max = shape_of_box(box_[max_index],boxs_default[max_index])
    #                 start_point_i,end_point_i = shape_of_box(box_[box_i],boxs_default[box_i])
    #                 inter = iou(start_point_max,end_point_max,start_point_i,end_point_i)
    #                 if inter>overlap:
    #                     classes_confidence[class_i][box_i] = 0
    #                     classes_confidence[class_i][max_index] = 0
    #                     result = [class_i,max_index,start_point_max[0],start_point_max[1],end_point_max[0],end_point_max[1]]
    #                     box_result.append(result)
    # return box_result

def draw_after_nms(epoch,windowname, pred_confidence, pred_box, image_, boxs_default):
    image = np.transpose(image_, (1, 2, 0)).astype(np.uint8)
    image1 = np.zeros(image.shape, np.uint8)
    image2 = np.zeros(image.shape, np.uint8)


    image1[:] = image[:]
    image2[:] = image[:]
    results = non_maximum_suppression(pred_confidence,pred_box,boxs_default,0.5,0.5)
    # print("result in draw_after_nms is ")
    # print(results)
    for result in results:
        cat = result[0]
        index = result[1]
        start_point = (result[2],result[3])
        end_point = (result[4],result[5])
        cv2.rectangle(image1,start_point,end_point,color=colors[cat],thickness=2)
        default_start,default_end = start_end_point(boxs_default[index][0],boxs_default[index][1],boxs_default[index][2],boxs_default[index][3])
        cv2.rectangle(image2,default_start,default_end,color=colors[cat],thickness=2)
    h, w, _ = image1.shape
    image = np.zeros([h , w * 2, 3], np.uint8)
    image[:h, :w] = image1
    image[:h, w:] = image2
    cv2.imwrite('/home/zbc/Visual Computing/Assignment3-Py-Version/Assignment3-Py-Version/CMPT733-Lab3-Workspace/nms/'+str(epoch)+'nms.png',image)

    #return img1,img2
    #TODO: non maximum suppression














