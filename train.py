from core.dataset import Dataset
import tensorflow as tf
from core.cfg import cfg
from core.mask_rcnn_model import Mask_RCNN
from core.mask_rcnn_model import rpn_class_loss_graph, rpn_bbox_loss_graph, mrcnn_class_loss_graph, \
    mrcnn_bbox_loss_graph, mrcnn_mask_loss_graph
tf.logging.set_verbosity(tf.logging.INFO)

def train_input_fn(mode='val', base_folder='voc\\val',
                   tfrecord_folder='data',
                   data_reload=True,
                   use_numpy_style=False):
    dataset_train = Dataset(mode=mode, base_folder=base_folder,
                            tfrecord_folder=tfrecord_folder,
                            data_reload=data_reload,
                            use_numpy_style=use_numpy_style)
    return dataset_train.batch_data


def val_input_fn(mode='val', base_folder='voc\\val',
                 tfrecord_folder='data',
                 data_reload=True,
                 use_numpy_style=False):
    dataset_val = Dataset(mode=mode, base_folder=base_folder,
                          tfrecord_folder=tfrecord_folder,
                          data_reload=data_reload,
                          use_numpy_style=use_numpy_style)
    return dataset_val.batch_data


def model_fn(features, labels, mode, params):
    gt_boxes = labels[0]
    gt_masks = labels[1]
    gt_activate_ids = labels[2]
    gt_rpn_match = labels[3]
    gt_rpn_bbox = labels[4]
    gt_class_ids = labels[5]

    rpn_class_logits, rpn_bbox, mrcnn_class_logits, mrcnn_bbox, mrcnn_mask, target_class_ids, target_bbox, target_mask = Mask_RCNN(
        mode=mode).build(input_image=features,
                         input_gt_class_ids=gt_activate_ids,
                         input_gt_box=gt_boxes,
                         input_gt_masks=gt_masks)

    if mode == tf.estimator.ModeKeys.PREDICT:  # 预测处理
        return tf.estimator.EstimatorSpec(mode, predictions=[rpn_class_logits, rpn_bbox, mrcnn_class_logits, mrcnn_bbox,
                                                             mrcnn_mask, target_class_ids])

    # 计算loss
    rpn_class_loss = rpn_class_loss_graph(gt_rpn_match, rpn_class_logits)

    rpn_bbox_loss = rpn_bbox_loss_graph(cfg.BATCH_SIZE, gt_rpn_bbox, gt_rpn_match, rpn_bbox)

    class_loss = mrcnn_class_loss_graph(cfg.NUM_CLASS, cfg.BATCH_SIZE, target_class_ids, mrcnn_class_logits,
                                        gt_class_ids)

    bbox_loss = mrcnn_bbox_loss_graph(target_bbox, target_class_ids, mrcnn_bbox)

    mask_loss = mrcnn_mask_loss_graph(target_mask, target_class_ids, mrcnn_mask)

    loss = cfg.LOSS_WEIGHTS['rpn_class_loss'] * rpn_class_loss + cfg.LOSS_WEIGHTS['rpn_class_loss'] * rpn_bbox_loss + \
           cfg.LOSS_WEIGHTS['rpn_class_loss'] * class_loss + cfg.LOSS_WEIGHTS['rpn_class_loss'] * bbox_loss + mask_loss


    if mode == tf.estimator.ModeKeys.EVAL:  # 测试处理
        meanloss = tf.metrics.mean(loss)  # 添加评估输出项
        metrics_op = {'meanloss': meanloss}
    else:
        metrics_op=None

    # 训练处理.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    #控制台输出日志
    tensors_to_log = {'rpn_class_loss': rpn_class_loss,
                      'rpn_bbox_loss': rpn_bbox_loss,
                      'class_loss': class_loss,
                      'bbox_loss': bbox_loss,
                      'mask_loss': mask_loss,
                      'loss':loss}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1)
    return tf.estimator.EstimatorSpec(mode, loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=metrics_op,
                                      training_hooks=[logging_hook])


tf.logging.set_verbosity(tf.logging.INFO)  # 能够控制输出信息  ，
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  # 构建gpu_options，防止显存占满
session_config = tf.ConfigProto(gpu_options=gpu_options)
# 构建估算器
estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir='./estimator_hook', params={'learning_rate': 0.0000001},
                                   config=tf.estimator.RunConfig(session_config=session_config))


for i in range(100):
    print('第', str(i), '个epoch')
    estimator.train(lambda: train_input_fn(mode='val', base_folder='voc\\val',
                                           tfrecord_folder='data',
                                           data_reload=True,
                                           use_numpy_style=False))
tf.logging.info("训练完成.")  # 输出训练完成
