import os
import cv2
'''
主要目的，解析出image，image_meta（可以不用），gt_class_ids，gt_boxes，gt_masks
'''
img_path=r'E:\Pycharm_project\mask_rcnn_TF\voc\train\imgs\2007_000032.jpg'
label_path = r'E:\Pycharm_project\mask_rcnn_TF\voc\train\labels\2007_000032.png'
img = cv2.imread(img_path,1)
label = cv2.imread(label_path,0)
