import tensorflow as tf
import numpy as np
import cv2

import yolo_network


tf.enable_eager_execution()
tf.executing_eagerly()


imgs = cv2.imread(
    '../../../../../微血管瘤分块数据/6x6/image/11818_right_0.JPEG')
imgs = tf.convert_to_tensor(cv2.resize(imgs, (256, 256)), dtype=tf.float32)
imgs = tf.reshape(imgs, [-1, 256, 256, 3])

logits = yolo_network.DarkNet()(
    imgs=tf.convert_to_tensor(imgs, dtype=tf.float32),
    first_conv_filters=32,
    dark_base_filters_list=[64, 128, 256, 512, 1024],
    residual_nums_per_base_layer=[1, 2, 8, 8, 4],
    conv_kernels_for_combined_features=[(3, 3)],
    filters_for_combined_features=[512],
    grids=[8, 8],
    prior_num_per_cell=5,
    class_num=1,
    training=True
)

logits_numpy = logits.numpy()
print(np.where(np.isnan(logits)))
