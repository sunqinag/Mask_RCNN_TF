import numpy as np
import tensorflow as tf

from . import feature_extractor
from . import layers


class DarkNet(feature_extractor.IFeatureExtractor):
    def __init__(self, **feature_extractor_params):
        '''
        first_conv_filters:
            darknet中第一个卷积层的filter个数
        dark_base_filter_list:
            每个dark base layer输出的channel个数
        residual_num_per_base_layer:
            每个dark base layer中残差结构的重复个数
        activation:
            激活函数
        '''
        super().__init__()
        self.first_conv_filters = feature_extractor_params[
            'first_conv_filters']
        self.dark_base_filter_list = feature_extractor_params[
            'dark_base_filter_list']
        self.residual_nums_per_base_layer = feature_extractor_params[
            'residual_nums_per_base_layer']
        self.activation = feature_extractor_params['activation']

    def __call__(self, imgs, training):
        '''
        imgs:
            输入网络的img batch
        training:
            是否是在训练

        返回值:
            最后一个feature map
        '''
        if len(self.dark_base_filter_list) != len(
                self.residual_nums_per_base_layer):
            raise ValueError(
                'dark_base_filters_list与residual_num_per_base_layer个数不符')
        self.dark_base_ind = 0
        last_layer = layers.conv_layer_with_batch_normalization(
            imgs,
            self.first_conv_filters,
            3,
            1,
            'same',
            self.activation,
            training
        )
        self.feature_maps.append(last_layer)
        for filters, residual_layer_num in zip(
            self.dark_base_filter_list,
            self.residual_nums_per_base_layer
        ):
            last_layer = self.__dark_base_layer(
                last_layer, filters, residual_layer_num, training)
            self._feature_maps.append(last_layer)
        return last_layer

    def __dark_base_layer(self, inputs, filters, residual_num, training):
        '''
        inputs:
            输入
        filters:
            一个darknet基础结构的残差后的输出的通道数
        residual_num:
            每个base layer中残差结构的个数
        training:
            是否是在训练

        返回值:
            每个dark base layer卷积核得到的tensor
        '''
        with tf.name_scope('darknet_base_layer'):
            with tf.variable_scope('darknet_base_layer_' + str(self.dark_base_ind)):
                last_layer = layers.conv_layer_with_batch_normalization(
                inputs, filters, 3, 2, 'same', self.activation, training)
                for k in range(residual_num):
                    with tf.name_scope('res'):
                        conv_1x1_1 = layers.conv_layer_with_batch_normalization(
                            last_layer,
                            int(np.floor(filters / 2)),
                            kernel_size=1,
                            strides=1,
                            padding='same',
                            activation=self.activation,
                            training=training)
                        conv_3x3_1 = layers.conv_layer_with_batch_normalization(
                            conv_1x1_1, filters, 3, 1, 'same', None, training)
                        last_layer = self.activation(conv_3x3_1 + last_layer)
                self.dark_base_ind += 1
            return last_layer
