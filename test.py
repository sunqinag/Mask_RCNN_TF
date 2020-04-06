import cv2, os
import numpy as np
import tensorflow as tf
from core.mask_rcnn_model import Mask_RCNN
from core.dataset import Dataset
import matplotlib.pyplot as plt

# tf.enable_eager_execution()

rpn_visualize_dir = 'rpn_feature_image'

dataset = Dataset(mode='val', base_folder=r'D:\Pycharm_Project\Mask_RCNN_TF\voc\val',
                  tfrecord_folder=r'D:\Pycharm_Project\Mask_RCNN_TF\data')
input_image, input_boxes, input_masks, input_class_ids = dataset.image, dataset.boxes, dataset.masks, dataset.class_ids

# input_image = tf.convert_to_tensor(input_image)
# 调试模式
# image = tf.convert_to_tensor(image)
# image = tf.cast(image, tf.float32)
# image = tf.concat([image,image],axis=0)
# model = Mask_RCNN(mode='training',
#                   input_rpn_match=None,
#                   input_rpn_bbox=None).build(input_image=image)


# 拿图模式
model = Mask_RCNN(mode='training',
                  input_rpn_match=None,
                  input_rpn_bbox=None).build(input_image=input_image,
                                             input_gt_class_ids=input_class_ids,
                                             input_gt_box=input_boxes,
                                             input_gt_masks=input_masks,
                                             )
