# coding: utf-8
from __future__ import division, print_function
import tensorflow as tf
from misc_utils import load_weights
from model.darknet import DarkNet

num_class = 80
img_size = 416
weight_path = './data/darknet_weights/yolov3.weights'
save_path = './data/darknet_weights/darknet.ckpt'

with tf.Session() as sess:
    inputs = tf.placeholder(tf.float32, [None, img_size, img_size, 3])

    darknet_feature_extractor = DarkNet(
            first_conv_filters=32,
            dark_base_filter_list=[64, 128, 256, 512, 1024],
            residual_nums_per_base_layer=[1, 2, 8, 8, 4],
            activation=tf.nn.leaky_relu
        )


    last_layer, varss =  darknet_feature_extractor(inputs, training=None)

    saver = tf.train.Saver(var_list=tf.global_variables())
    load_ops = load_weights(tf.global_variables(), weight_path)
    sess.run(load_ops)
    saver.save(sess, save_path=save_path)
    print('TensorFlow model checkpoint has been saved to {}'.format(save_path))



