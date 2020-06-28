import tensorflow as tf
import cv2, os
from core.dataset import Dataset
from core.mask_rcnn_model import rpn_class_loss_graph, rpn_bbox_loss_graph, mrcnn_class_loss_graph, \
    mrcnn_bbox_loss_graph, mrcnn_mask_loss_graph
from core.dataset import Dataset
import tensorflow as tf
from core.cfg import cfg
from core.mask_rcnn_model import Mask_RCNN

tf.enable_eager_execution()


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
    # gt_rpn_match = tf.cast(gt_rpn_match,tf.int32)
    # rpn_class_logits=tf.cast(rpn_class_logits,tf.int32)
    rpn_class_loss = rpn_class_loss_graph(gt_rpn_match, rpn_class_logits)
    print('rpn_class_loss:', rpn_class_loss)
    rpn_bbox_loss = rpn_bbox_loss_graph(cfg.BATCH_SIZE, gt_rpn_bbox, gt_rpn_match, rpn_bbox)
    print('rpn_bbox_loss：', rpn_bbox_loss)
    # target_class_ids=tf.cast(target_class_ids,tf.int32)
    # mrcnn_class_logits=tf.cast(mrcnn_class_logits,tf.int32)
    class_loss = mrcnn_class_loss_graph(cfg.NUM_CLASS, cfg.BATCH_SIZE, target_class_ids, mrcnn_class_logits,
                                        gt_class_ids)
    print('class_loss:', class_loss)
    class_loss=0
    bbox_loss = mrcnn_bbox_loss_graph(target_bbox, target_class_ids, mrcnn_bbox)
    print('bbox_loss:', bbox_loss)
    mask_loss = mrcnn_mask_loss_graph(target_mask, target_class_ids, mrcnn_mask)
    print('mask_loss:', mask_loss)
    loss = cfg.LOSS_WEIGHTS['rpn_class_loss'] * rpn_class_loss + cfg.LOSS_WEIGHTS['rpn_class_loss'] * rpn_bbox_loss + \
           cfg.LOSS_WEIGHTS['rpn_class_loss'] * class_loss + cfg.LOSS_WEIGHTS['rpn_class_loss'] * bbox_loss + mask_loss
    print('loss:', loss)

    # 训练处理.
    # assert mode == tf.estimator.ModeKeys.TRAIN
    # optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    # train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return 0


dataset = Dataset(mode='val', base_folder='./voc/val',
                  tfrecord_folder='./data',
                  data_reload=True,
                  use_numpy_style=False)
for i in range(10000):
    (image, (boxes, masks, class_ids, rpn_match, rpn_bbox, activate_ids)) = dataset.batch_data
    model_fn(features=image, labels=(boxes, masks, class_ids, rpn_match, rpn_bbox, activate_ids), mode='train',params={'learning_rate':0.001})
    print('============================')