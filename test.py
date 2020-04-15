import cv2, os
import numpy as np
from core.cfg import cfg
import tensorflow as tf
from core.mask_rcnn_model import Mask_RCNN,generate_rpn_match_and_rpn_bbox
from core.dataset import Dataset
import matplotlib.pyplot as plt

# tf.enable_eager_execution()

rpn_visualize_dir = 'rpn_feature_image'

dataset = Dataset(mode='val', base_folder='voc/val',
                  tfrecord_folder='data', use_numpy_style=False)
input_image, input_boxes, input_masks, input_class_ids = dataset.image, dataset.boxes, dataset.masks, dataset.class_ids

print('input_image shape:', input_image.shape)
print('input_boxes shape:', input_boxes.shape)
print('input_masks shape:', input_masks.shape)
print('input_class_ids shape:', input_class_ids.shape)

dataset2 = Dataset(mode='val', base_folder='voc/val',
                   tfrecord_folder='data', use_numpy_style=True)
batch_image, batch_mask, batch_bbox, batch_class_ids = dataset2.batch_image, dataset2.batch_mask, dataset2.batch_bbox, dataset2.batch_class_ids

print('=====================================================================')
print('batch_image shape:', batch_image.shape)
print('batch_mask shape:', batch_mask.shape)
print('batch_bbox shape:', batch_bbox.shape)
print('batch_class_ids shape:', batch_class_ids.shape)

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
                  input_rpn_bbox=None)
# model.build(input_image=input_image,
#             input_gt_class_ids=input_class_ids,
#             input_gt_box=input_boxes,
#             input_gt_masks=input_masks,
#             )
batch_rpn_match,batch_rpn_bbox=generate_rpn_match_and_rpn_bbox(batch_class_ids,batch_bbox)
