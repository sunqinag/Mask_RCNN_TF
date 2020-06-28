import tensorflow as tf

from .model import yolo_network
from .loss import losses
from .model.feature_extractor import IFeatureExtractor


class YOLOEstimator(tf.estimator.Estimator):
    def __init__(self,
                 model_dir,
                 feature_extractor,
                 conv_kernels_for_combined_features,
                 filters_for_combined_features,
                 grids,
                 prior_num_per_cell,
                 combined_feature_map_inds,
                 class_num,
                 activation,
                 output_layer_type,
                 config=None,
                 params=None,
                 warm_start_from=None):
        '''
        model_dir:
            checkpoint存储路径
        feature_extractor:
            用于提取特征的IFeatureExtractor实例
        conv_kernels_for_combined_features:
            对多尺度融合的feature map进行卷积的卷积核的list
        filters_for_combined_features:
            对多尺度融合的feature map进行卷积的卷积层输出通道数
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
        config:
            estimator运行的配置
        params:
            模型训练需要的超参，包括：
            coord_weight——bbox回归loss的权重
            obj_weight——objectness loss的权重
            cls_weight——分类loss的权重
            learning_rate——学习率
        warm_start_from:
            模型加载的checkpoint的路径
        '''
        super().__init__(self.__model_fn,
                         model_dir,
                         config,
                         params,
                         warm_start_from)
        if not isinstance(feature_extractor, IFeatureExtractor):
            raise ValueError('feature_extractor不是IFeatureExtractor的实例')
        self.feature_extractor = feature_extractor
        self.conv_kernels_for_combined_features = \
            conv_kernels_for_combined_features
        self.filters_for_combined_features = filters_for_combined_features
        self.grids = grids
        self.prior_num_per_cell = prior_num_per_cell
        self.combined_feature_map_inds = combined_feature_map_inds
        self.class_num = class_num
        self.activation = activation
        self.output_layer_type = output_layer_type

    def __model_fn(self, features, labels, mode, params, config):
        training = mode == tf.estimator.ModeKeys.TRAIN
        imgs = features['img']

        logits = yolo_network.YoloNetwork(self.feature_extractor)(
            imgs,
            self.conv_kernels_for_combined_features,
            self.filters_for_combined_features,
            self.grids,
            self.prior_num_per_cell,
            self.combined_feature_map_inds,
            self.class_num,
            self.activation,
            self.output_layer_type,
            training)
        objectness_logits = logits[:, :, :, :, 0]
        bbox_logits = logits[:, :, :, :, 1: 5]
        class_logits = logits[:, :, :, :, 5:]

        if (mode == tf.estimator.ModeKeys.TRAIN or
                mode == tf.estimator.ModeKeys.EVAL):
            objectness_label = labels[:, :, :, :, 0]
            obj_loss_calc_label = labels[:, :, :, :, 1]
            bboxes_label = labels[:, :, :, :, 2: 6]
            class_label = labels[:, :, :, :, 6:]

            loss_dict = losses.YoloLoss(
                self.grids,
                params['coord_weight'],
                params['obj_weight'],
                params['noobj_weight'],
                params['cls_weight']
            )(
                bbox_logits,
                objectness_logits,
                class_logits,
                bboxes_label,
                objectness_label,
                class_label,
                obj_loss_calc_label
            )
            loss = loss_dict['loss']
        else:
            loss = None
            loss_dict = {}

        predictions = {'objectness': tf.nn.sigmoid(objectness_logits),
                       'bbox_t': bbox_logits,
                       'obj_class': tf.nn.sigmoid(class_logits)}

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(
                params['learning_rate']).minimize(
                    loss, tf.train.get_global_step())
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.group([train_op, update_ops])
        else:
            train_op = None

        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {
                'bbox_error': tf.metrics.mean_squared_error(
                    bboxes_label, bbox_logits),
                'precision': tf.metrics.precision(
                    class_label, tf.nn.sigmoid(class_logits)),
                'recall': tf.metrics.recall(
                    class_label, tf.nn.sigmoid(class_logits))
            }
        else:
            eval_metric_ops = None

        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {
                (tf.saved_model.signature_constants.
                 DEFAULT_SERVING_SIGNATURE_DEF_KEY):
                tf.estimator.export.PredictOutput({
                    'objectness': tf.nn.sigmoid(objectness_logits),
                    'bbox_logits': bbox_logits,
                    'class_prob': tf.nn.sigmoid(class_logits)})
            }
        else:
            export_outputs = None

        logging_hook = tf.train.LoggingTensorHook(
            tensors=loss_dict,
            every_n_iter=10
        )

        return tf.estimator.EstimatorSpec(
            mode,
            predictions,
            loss,
            train_op,
            eval_metric_ops,
            export_outputs,
            training_hooks=[logging_hook]
        )
