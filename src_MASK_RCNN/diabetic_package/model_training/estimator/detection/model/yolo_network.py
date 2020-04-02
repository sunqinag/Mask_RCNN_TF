import tensorflow as tf

from . import layers


class YoloNetwork():
    def __init__(self, feature_extractor):
        '''
        feature_extractor:
            用于提取特征的IFeatureExtractor实例
        '''
        self.feature_extractor = feature_extractor

    def __call__(self,
                 imgs,
                 conv_kernels_for_combined_features,
                 filters_for_combined_features,
                 grids,
                 prior_num_per_cell,
                 combined_feature_map_inds,
                 class_num,
                 activation,
                 output_layer_type,
                 training):
        '''
        imgs:
            输入网络的img batch
        conv_kernels_for_combined_features:
            用于拼接的多尺度特征的卷积核尺寸的list
        filters_for_combined_features:
            用于拼接的多尺度特征的卷积层的卷积核个数的list
        grids:
            分块数目
        prior_num_per_cell:
            每个grid cell prior的数量
        combined_feature_map_inds:
            feature map extractor返回的所有尺度的feature map中用于预测的
            feature map的索引，combined feature map都会被resize到
            combined_feature_map_inds指定的最后一维feature map的大小
        class_num:
            objectness的分类个数
        activation:
            激活函数
        output_layer_type:
            输出层的处理方式，包括:
                global_average_pooling, global_average_pooling_conn, conv
        training:
            是否是在训练

        返回值:
            返回[batch_size, grid_y, grid_x, prior_num_per_cell, 5 + class_num]
        '''
        self.conv_kernels_for_combined_features = \
            conv_kernels_for_combined_features
        self.filters_for_combined_features = filters_for_combined_features,
        self.grids = grids
        self.prior_num_per_cell = prior_num_per_cell
        self.combined_feature_map_inds = combined_feature_map_inds
        self.class_num = class_num
        self.activation = activation
        self.training = training
        _ = self.feature_extractor(imgs, training)
        resized_shape = self.feature_extractor.feature_maps[
            self.combined_feature_map_inds[-1]].shape[1: 3]
        combined_feature_map = []
        for k in combined_feature_map_inds:
            combined_feature_map.append(
                tf.image.resize_bilinear(
                    self.feature_extractor.feature_maps[k],
                    resized_shape
                )
            )
        last_layer = tf.concat(combined_feature_map, axis=-1)
        for kernel_size, filters in zip(conv_kernels_for_combined_features,
                                        filters_for_combined_features):
            last_layer = layers.conv_layer_with_batch_normalization(
                last_layer,
                filters,
                kernel_size,
                1,
                'same',
                activation,
                training)
        if output_layer_type == 'global_average_pooling':
            return self.__global_average_pooling_output(last_layer)
        elif output_layer_type == 'global_average_pooling_conn':
            return self.__global_average_pooling_conn_output(last_layer)
        elif output_layer_type == 'conv':
            return self.__conv_output(last_layer)
        else:
            raise ValueError('output_layer_type的输入值是不支持的输出层类型')

    def __global_average_pooling_output(self, conv_combined_feature_map):
        logits = layers.conv_layer_with_batch_normalization(
            conv_combined_feature_map,
            self.grids[0] * self.grids[1] * self.prior_num_per_cell * (
                5 + self.class_num),
            1,
            1,
            'same',
            None,
            self.training
        )
        logits = tf.reduce_mean(logits, axis=[1, 2])
        logits = tf.reshape(
            logits,
            [-1,
             self.grids[0],
             self.grids[1],
             self.prior_num_per_cell,
             (5 + self.class_num)]
        )
        return logits

    def __global_average_pooling_conn_output(self, conv_combined_feature_map):
        conv = layers.conv_layer_with_batch_normalization(
            conv_combined_feature_map,
            self.grids[0] * self.grids[1] * self.prior_num_per_cell * (
                5 + self.class_num),
            1,
            1,
            'same',
            self.activation,
            self.training
        )
        logits = tf.layers.dense(
            conv,
            self.grids[0] * self.grids[1] * self.prior_num_per_cell * (
                5 + self.class_num),
            activation=None
        )
        logits = tf.layers.batch_normalization(logits, self.training)
        return logits

    def __conv_output(self, conv_combined_feature_map):
        conv_3x3 = layers.conv_layer_with_batch_normalization(
            conv_combined_feature_map,
            self.prior_num_per_cell * (5 + self.class_num),
            3,
            1,
            'same',
            self.activation,
            self.training
        )
        logits = layers.conv_layer_with_batch_normalization(
            conv_3x3,
            self.prior_num_per_cell * (5 + self.class_num),
            1,
            1,
            'same',
            None,
            self.training
        )
        return tf.reshape(
            logits,
            [-1,
             logits.shape[1],
             logits.shape[2],
             self.prior_num_per_cell,
             5 + self.class_num]
        )
