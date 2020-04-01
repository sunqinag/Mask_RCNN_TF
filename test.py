import cv2, os
import numpy as np
import tensorflow as tf
from core.mask_rcnn_model import Mask_RCNN
from core.dataset import Dataset
import matplotlib.pyplot as plt

tf.enable_eager_execution()

rpn_visualize_dir = 'rpn_feature_image'

# input_image = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
dataset = Dataset(mode='val', base_folder=r'E:\Pycharm_project\mask_rcnn_TF\voc\val', tfrecord_folder='',
                      data_reload=True)
input_image, input_boxes, input_masks, input_class_ids = dataset.image, dataset.boxes, dataset.masks, dataset.class_ids


input_image = tf.convert_to_tensor(input_image)
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
                                             gt_boxes=input_boxes,
                                             input_gt_masks=input_masks,
                                             )

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     rois = sess.run(model, feed_dict={
#         input_image: image
#     })
# 拿到proposal层结果看看标框
d = 0
# 可视化P2, P3, P4, P5, P6层返回特征
# for i, rpn in enumerate(rpn_feature_maps):
#     rpn = np.squeeze(rpn).astype(np.uint8)
#     second_class_dir = rpn_visualize_dir+os.sep+'P' + str(i+2)
#     if not os.path.exists(second_class_dir):
#         os.mkdir(second_class_dir)
#     for j in range(rpn.shape[-1]):
#         plt.figure(figsize=(6,6))
#         plt.imshow(rpn[:,:,j])
#         # plt.savefig(rpn_visualize_dir + os.sep + 'P' + str(i) +'_ch'+str(j)+ '_image')
#         plt.show()
#         cv2.imwrite(second_class_dir+os.sep +'ch'+str(j)+ '_image.png', rpn[:,:,j])
#
