import cv2,os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

CLASS_NUM = 10
LEARNING_RATE = 0.001

batch_size = 3
train_iters = 20000
display_step = 5

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


class ResNet50:
    def __init__(self, regulizer_scale, istraning=True):
        self.num_block = [3, 8, 36, 3]
        self.istraining = istraning
        self.regulizer_scale = regulizer_scale

    def regularizer(self, weight):
        return tf.contrib.layers.l1_l2_regularizer(self.regulizer_scale[0], self.regulizer_scale[1])(weight)

    def block(self, input_data, filter_num, strides):
        '''双层3*3卷积的block'''
        data = input_data
        input_data = tf.layers.conv2d(inputs=input_data, filters=filter_num[0], strides=strides[0], kernel_size=1,
                                      padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer,
                                      kernel_regularizer=self.regularizer)
        input_data = tf.layers.batch_normalization(inputs=input_data)
        input_data = tf.layers.conv2d(inputs=input_data, filters=filter_num[0], strides=strides[1], kernel_size=3,
                                      padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer,
                                      kernel_regularizer=self.regularizer)
        input_data = tf.layers.batch_normalization(inputs=input_data)
        input_data = tf.layers.conv2d(inputs=input_data, filters=filter_num[1], strides=strides[1], kernel_size=1,
                                      padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer,
                                      kernel_regularizer=self.regularizer)
        input_data = tf.layers.batch_normalization(inputs=input_data)

        # 对输入数据做处理保证与通过三层卷积的结构尺度和通道数相同
        if strides[0] == 1:
            data = tf.layers.conv2d(inputs=data, filters=filter_num[1], kernel_size=1, padding='SAME',
                                    activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer,
                                    kernel_regularizer=self.regularizer)
        else:
            data = tf.layers.conv2d(inputs=data, filters=filter_num[1], strides=strides[0], kernel_size=1,
                                    padding='SAME',
                                    activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer,
                                    kernel_regularizer=self.regularizer)
        # 跨层连接
        input_data = tf.add(data, input_data)
        return input_data

    def build(self, x):
        input_data = tf.reshape(x, [-1, 28, 28, 1])
        with tf.variable_scope('conv1_x'):
            input_data = tf.layers.conv2d(inputs=input_data, filters=64, kernel_size=7,
                                          strides=1,
                                          padding='SAME', activation=tf.nn.relu,
                                          kernel_initializer=tf.truncated_normal_initializer,
                                          kernel_regularizer=self.regularizer)
            # input_data = tf.layers.max_pooling2d(inputs=input_data, pool_size=3, strides=2)
        # 第一部分
        with tf.variable_scope('conv2_x'):
            input_data = self.block(input_data, [64, 256], [1, 1])
            for i in range(2):
                input_data = self.block(input_data, [64, 256], [1, 1])

        with tf.variable_scope('conv3_x'):
            input_data = self.block(input_data, [128, 512], [1, 1])
            for i in range(7):
                input_data = self.block(input_data, [128, 512], [1, 1])

        with tf.variable_scope('conv4_x'):
            input_data = self.block(input_data, [256, 1024], [1, 1])
            for i in range(35):
                input_data = self.block(input_data, [256, 1024], [1, 1])

        with tf.variable_scope('conv5_x'):
            input_data = self.block(input_data, [512, 2048], [1, 1])
            for i in range(2):
                input_data = self.block(input_data, [256, 1024], [1, 1])

        input_data = tf.layers.average_pooling2d(inputs=input_data, pool_size=3, strides=2)
        input_data = tf.layers.flatten(input_data)
        out = tf.layers.dense(inputs=input_data, units=1000)
        out = tf.layers.dropout(out,0.7)
        out = tf.layers.dense(out, CLASS_NUM)

        return out


logit = ResNet50(regulizer_scale=[0.0001, 0.0001]).build(x)

# 损失函数
loss = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y)

# 优化器
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

# 评估函数
correct_pred = tf.equal(tf.argmax(logit, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化全局变量
init = tf.global_variables_initializer()

def set_config():
    # 控制使用率
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # 假如有16GB的显存并使用其中的8GB:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config = tf.ConfigProto(gpu_options=gpu_options)
    # session = tf.Session(config=config)
    return config


cfg=set_config()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step < train_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # Alex_net(batch_x,weights,biases,0.7)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            loss_, acc_ = sess.run([loss, accuracy], feed_dict={
                x: batch_x,
                y: batch_y,
            })
            print('第', str(step * batch_size), "次 MiniBatch_loss:", loss_, '准确率accuracy：', acc_)
        step += 1
    print('Optimiz Finished')
    saver = tf.saver
    # 计算测试机精度
    print("测试集精度为：", sess.run([loss, accuracy], feed_dict={
        x: mnist.test.images[:256],
        y: mnist.test.labels[:256],
    }))


