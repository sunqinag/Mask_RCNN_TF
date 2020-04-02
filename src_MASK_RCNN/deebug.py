import math
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from core.dataset import Dataset


mask_rcnn_model = __import__("8-29  mask_rcnn_model")
MaskRCNN = mask_rcnn_model.MaskRCNN
utils = __import__("8-30  mask_rcnn_utils")
visualize = __import__("8-32  mask_rcnn_visualize")


tf.enable_eager_execution()

rpn_visualize_dir = 'rpn_feature_image'

dataset = Dataset(mode='val', base_folder=r'E:\Pycharm_project\mask_rcnn_TF\voc\val',
                  tfrecord_folder=r'E:\Pycharm_project\mask_rcnn_TF\data')
input_image, input_boxes, input_masks, input_class_ids = dataset.image, dataset.boxes, dataset.masks, dataset.class_ids
model = MaskRCNN(mode='training',
                 model_dir='log',
                 num_class=21,
                 batch_size=3).build(input_image=input_image,
                                             input_gt_class_ids=input_class_ids,
                                             input_gt_boxes=input_boxes,
                                             input_gt_masks=input_masks,
                                             )
