import tensorflow as tf
import core.net as net
from core.cfg import cfg
import core.utils as utils
import math
import numpy as np
from core.layers import DetectionTargetLayer, overlaps_graph


class Mask_RCNN:
    def __init__(self, mode):
        """
        mode: 可以是 "training" 或 "inference" 两种模式
        model_dir: 保存模型的路径
        """
        self.mode = mode
        self.batch_size = cfg.BATCH_SIZE
        self.num_class = 21

    def build(self, input_image, input_gt_class_ids, input_gt_box, input_gt_masks,
              input_gt_rpn_match, input_gt_rpn_bbox,input_activate_ids):  # 构建Mask R-CNN架构
        # input_image = tf.identity(input_image, name='input_image_layer')
        # input_image = tf.placeholder(shape=[None,cfg.IMAGE_MIN_DIM,cfg.IMAGE_MIN_DIM,3],name='input_image',dtype=tf.float32)
        # 检查尺寸合法性
        h, w = cfg.IMAGE_MIN_DIM, cfg.IMAGE_MIN_DIM
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("必须要被2的6次方整除.例如： 256, 320, 384, 448, 512, ... etc. ")
        if self.mode == 'training':  # 处理训练接口
            # 由图片和标注生成的关于锚点的值1 -1 0.用于RPN的标签
            # input_rpn_match = tf.placeholder(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            # 由图片和标注生成的关于锚点框的偏移量x,y 高宽缩放.用于RPN的标签
            # input_rpn_bbox = tf.placeholder(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)
            # 下面获得检测GT（class_id,bounding boxes,mask）
            # input_gt_class_ids = tf.placeholder(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # <editor-fold desc="Description">
            # 2. GT Boxes in pixels (zero padded)
            # </editor-fold>
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            # input_gt_boxes = tf.placeholder(
            #     shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # ===============================================分割线===================================================

            _, C2, C3, C4, C5 = net.resnet(input_image)

            # 特征金字塔层fpn。  最大的c1不要了。每层256个
            P5 = tf.layers.conv2d(inputs=C5, filters=256, kernel_size=2, padding='same', name='fpn_c5p5')
            P4_1 = tf.layers.conv2d_transpose(inputs=P5, filters=256, kernel_size=2, strides=(2, 2), padding='same',
                                              name="fpn_p5upsampled")
            P4_2 = tf.layers.conv2d(inputs=C4, filters=256, kernel_size=1, padding='same', name='fpn_c4p4')
            P4 = tf.add(P4_1, P4_2, name="fpn_p4add")

            P3_1 = tf.layers.conv2d_transpose(inputs=P4, filters=256, kernel_size=2, strides=(2, 2), padding='same',
                                              name="fpn_p4upsampled")
            P3_2 = tf.layers.conv2d(inputs=C3, filters=256, kernel_size=1, padding='same', name='fpn_c3p3')
            P3 = tf.add(P3_1, P3_2, name="fpn_p3add")

            P2_1 = tf.layers.conv2d_transpose(inputs=P3, filters=256, kernel_size=2, strides=(2, 2), padding='same',
                                              name="fpn_p2upsampled")
            P2_2 = tf.layers.conv2d(inputs=C2, filters=256, kernel_size=1, padding='same', name='fpn_c2p2')
            P2 = tf.add(P2_1, P2_2, name="fpn_p2add")

            # 再将融合后的网络进行一次卷积提取特征
            P2 = tf.layers.conv2d(inputs=P2, filters=256, kernel_size=3, padding='same', name='fpn_p2')
            P3 = tf.layers.conv2d(inputs=P3, filters=256, kernel_size=3, padding='same', name='fpn_p3')
            P4 = tf.layers.conv2d(inputs=P4, filters=256, kernel_size=3, padding='same', name='fpn_p4')
            P5 = tf.layers.conv2d(inputs=P5, filters=256, kernel_size=3, padding='same', name='fpn_p5')
            P6 = tf.layers.max_pooling2d(inputs=P5, pool_size=2, strides=(2, 2), name='fpn_p6')

            rpn_feature_maps = [P2, P3, P4, P5, P6]  # 用于rpn使用,P5是最全的特征，对p5进行下采样生成p6
            mrcnn_feature_maps = [P2, P3, P4, P5]  # 用于classifier heads 使用

            # ========================特征提取层之后进入rpn网络获得roi=======================================
            # 将rpn_feature_maps分别放入rpn中形成
            out = net.build_rpn_layer(rpn_feature_maps, len(cfg.RPN_ANCHOR_RATIOS), cfg.RPN_ANCHOR_STRIDE)
            # 将上面拿到的5个尺度的rpn_class_logits, rpn_probs, rpn_bbox结果Concatenate到一起
            rpn_class_logits = tf.concat([p[0] for p in out], axis=1)
            rpn_class_probs = tf.concat([p[1] for p in out], axis=1)
            rpn_bbox = tf.concat([p[2] for p in out], axis=1)

            # Generate proposals
            # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
            # and zero padded.
            # 需要保留ROI的个数
            proposal_count = cfg.POST_NMS_ROIS_TRAINING
            anchors = self.get_anchors([cfg.IMAGE_MAX_DIM, cfg.IMAGE_MAX_DIM, 3])  # 在一张图上画出众多候选框

            anchors = np.broadcast_to(anchors, (self.batch_size,) + anchors.shape)
            anchors = tf.convert_to_tensor(anchors)

            # 下一步进入roi筛选，返回nms去重后，前景分数最大的n个ROI,这里roi坐标取值实在0-1
            rpn_rois = net.ProposalLayer(proposal_count=proposal_count,
                                         nms_threshold=cfg.RPN_NMS_THRESHOLD,
                                         batch_size=self.batch_size)([rpn_class_probs, rpn_bbox, anchors])

            # =======================================================================
            # fpn网络对rpn_rois区域与特征数据 mrcnn_feature_maps进行计算。识别出分类、边框和掩码

            # 前面有个inference过程，但先不做，只做训练部分

            # 训练模式下，获得输入数据的类
            # active_class_ids = tf.placeholder(dtype=tf.int32, shape=[None, self.num_class], name='acivate_class_ids')

            # 正常训练模式
            target_rois = rpn_rois
            # 根据输入的样本，制作RPN网络的标签
            rois, target_class_ids, target_bbox, target_mask = DetectionTargetLayer(batch_size=self.batch_size,
                                                                                    name='mrcnn_detection')(
                [target_rois, input_gt_class_ids, input_gt_box, input_gt_masks])

            # 分类器
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = net.fpn_classifier_graph(rois, mrcnn_feature_maps,
                                                                                   cfg.POOL_SIZE, self.num_class,
                                                                                   self.batch_size, train_bn=False,
                                                                                   # 不用bn
                                                                                   fc_layers_size=1024)  # 全连接层1024个节点
            # 进行语义分割，掩码预测
            mrcnn_mask = net.build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                                  cfg.MASK_POOL_SIZE, self.num_class, self.batch_size, train_bn=False)
            # 计算loss
            rpn_class_loss = rpn_class_loss_graph(input_gt_rpn_match, rpn_class_logits)
            rpn_bbox_loss = rpn_bbox_loss_graph(cfg.BATCH_SIZE, input_gt_rpn_bbox, input_gt_rpn_match, rpn_bbox)
            class_loss = mrcnn_class_loss_graph(self.num_class,self.batch_size,target_class_ids, mrcnn_class_logits, input_gt_class_ids)
            bbox_loss =mrcnn_bbox_loss_graph(target_bbox, target_class_ids, mrcnn_bbox)
            mask_loss =mrcnn_mask_loss_graph(target_mask, target_class_ids, mrcnn_mask)






    def get_anchors(self, image_shape):
        '''根据指定图片大小生成锚点.输入为原图尺寸'''
        backbone_shapes = compute_backbone_shapes(image_shape)
        # 缓存锚点
        if not hasattr(self, "_anchor_cache"):  # 用来判断是否存在某个属性。现在没有就给建立一个self属性
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # 生成锚点(这里坐标值得范围是0-1024)
            a = utils.generate_pyramid_anchors(cfg.RPN_ANCHOR_SCALES, cfg.RPN_ANCHOR_RATIOS,
                                               backbone_shapes, cfg.BACKBONE_STRIDES, cfg.RPN_ANCHOR_STRIDE)
            self.anchors = a
            # 设为标准坐标(转化后坐标值取值范围为0-1)
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]  # 现在这里的坐标值是归一化的，还有负值

    def get_anchors_tf(self, image_shape):
        '''根据指定图片大小生成锚点.输入为原图尺寸'''
        backbone_shapes = compute_backbone_shapes(image_shape)
        # 缓存锚点
        if not hasattr(self, "_anchor_cache"):  # 用来判断是否存在某个属性。现在没有就给建立一个self属性
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # 生成锚点(这里坐标值得范围是0-1024)
            a = utils.generate_pyramid_anchors(cfg.RPN_ANCHOR_SCALES, cfg.RPN_ANCHOR_RATIOS,
                                               backbone_shapes, cfg.BACKBONE_STRIDES, cfg.RPN_ANCHOR_STRIDE)
            self.anchors = a
            # 设为标准坐标(转化后坐标值取值范围为0-1)
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]  # 现在这里的坐标值是归一化的，还有负值

def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = tf.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = tf.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = tf.cond(tf.size(y_true) > 0,
                    lambda:tf.nn.softmax_cross_entropy_with_logits(target=y_true, output=y_pred),
                    lambda:tf.constant(0.0))
    loss = tf.reduce_mean(loss)
    return loss




def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    target_bbox = tf.reshape(target_bbox, (-1, 4))
    print("___________pred_bbox________", pred_bbox)
    pred_bbox = tf.reshape(pred_bbox, (-1, tf.shape(pred_bbox)[2], 4))
    print("___________pred_bbox________", pred_bbox)

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = tf.cond(tf.size(target_bbox) > 0,
                    lambda :smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    lambda:tf.constant(0.0))
    loss = tf.reduce_mean(loss)
    return loss

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss




def mrcnn_class_loss_graph(num_class, batch_size, target_class_ids, pred_class_logits,
                           active_class_ids):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """
    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.cast(target_class_ids, 'int64')
    print("pred_class_logits_________", pred_class_logits.get_shape())

    pred_class_logits = tf.reshape(pred_class_logits, (batch_size, -1, num_class))
    print("mrcnn_class_logits____", pred_class_logits.get_shape())

    # Find predictions of classes that are not in the dataset.
    # 查找不在数据集中的类的预测
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # 更新此行以使用批处理>1。现在，它假定批处理中的所有图像都具有相同的active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)
    pred_active = tf.cast(pred_active,tf.float32)
    # Loss （batch，32）
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # 消除不在图像的活动类别中的类别的预测的损失。
    loss = loss * pred_active

    # 计算loss平均值，仅适用有助于做出预测的预测loss以得到正确的平均值.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss




def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = tf.cast(tf.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(tf.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    # loss = K.sparse_categorical_crossentropy(target=anchor_class,
    #                                          output=rpn_class_logits,
    #                                          from_logits=True)
    # loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    # 计算交叉熵
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_class_logits,
                                                                         labels=anchor_class,
                                                                         name='rpn_class_loss'))
    loss = tf.cond(tf.size(loss) > 0, lambda: tf.reduce_mean(loss), lambda: tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(batch_size, target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = tf.squeeze(rpn_match, -1)
    indices = tf.where(tf.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    # 将目标边界框增量修剪为与rpn_bbox相同的长度
    batch_counts = tf.reduce_sum(tf.cast(tf.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts, batch_size)

    # TODO: use smooth_l1_loss() rather than reimplementing here
    #       to reduce code duplication
    diff = tf.abs(target_bbox - rpn_bbox)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)

    # loss = tf.switch(tf.size(loss) > 0, tf.mean(loss), tf.constant(0.0))
    loss = tf.cond(tf.size(loss) > 0, lambda: tf.reduce_mean(loss), lambda: tf.constant(0.0))
    return loss


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    根据值从x中的每一行选取不同数量的值
    # target_bbox, batch_counts, batch_size
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


# 有batch版本
def generate_rpn_match_and_rpn_bbox_with_batch(gt_class_ids, gt_boxes):
    # Anchors9
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = compute_backbone_shapes(
        [cfg.IMAGE_DIM, cfg.IMAGE_DIM])  # 【256 128 64 32 16】---1024/[4, 8, 16, 32, 64]
    anchors = utils.generate_pyramid_anchors(cfg.RPN_ANCHOR_SCALES,
                                             cfg.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             cfg.BACKBONE_STRIDES,
                                             cfg.RPN_ANCHOR_STRIDE)
    batch_rpn_match = np.zeros([cfg.BATCH_SIZE, anchors.shape[0], 1], dtype=np.int32)
    batch_rpn_bbox = np.zeros([cfg.BATCH_SIZE, cfg.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=np.int32)
    for i in range(cfg.BATCH_SIZE):
        rpn_match, rpn_bbox = build_rpn_targets(anchors, gt_class_ids[i, :], gt_boxes[i, :, :])
        batch_rpn_match[i] = rpn_match[:, np.newaxis]
        batch_rpn_bbox[i] = rpn_bbox
        unique = np.unique(rpn_match)
        print('unique:', unique)
    print('batch_rpn_match shape', batch_rpn_match.shape)
    print('batch_rpn_bbox shape', batch_rpn_bbox.shape)

    return batch_rpn_match, batch_rpn_bbox


# 无batch版本
def generate_rpn_match_and_rpn_bbox(gt_class_ids, gt_boxes):
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = compute_backbone_shapes(
        [cfg.IMAGE_DIM, cfg.IMAGE_DIM])  # 【256 128 64 32 16】---1024/[4, 8, 16, 32, 64]
    anchors = utils.generate_pyramid_anchors(cfg.RPN_ANCHOR_SCALES,
                                             cfg.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             cfg.BACKBONE_STRIDES,
                                             cfg.RPN_ANCHOR_STRIDE)
    rpn_match, rpn_bbox = build_rpn_targets(anchors, gt_class_ids, gt_boxes)

    unique = np.unique(rpn_match)
    print('unique:', unique)

    return rpn_match, rpn_bbox


def build_rpn_targets(anchors, gt_class_ids, gt_boxes):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral（iou在0.3和0.7之间）
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    # 其实rpn_bbox可以砍掉一般。因为只有放了一半的正锚点
    rpn_bbox = np.zeros((cfg.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]  每个值都是面积重叠的比例
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1  # 将Iou《0.3的设为-1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1  # 将锚点中，对应与任何一个bbox的Iou最大的值都设为1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1  # 将将Iou>= 0.7的设为
    # 要么充分重叠，要么不充分重叠。  介于二者之前的都是0

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    # 总共256个正负锚点框。正负锚点有超过半数的，通过随机值将其去掉。
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (cfg.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (cfg.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        # 计算bbox的高、宽、中心点
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w

        # 计算Anchor的高、宽、中心点
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        # 计算需要预测的中心点偏移比例及高、宽缩放比例
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= cfg.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox


# 计算resnet返回的形状
def compute_backbone_shapes(image_shape):
    returnshape = [[int(math.ceil(image_shape[0] / stride)),
                    int(math.ceil(image_shape[1] / stride))] for stride in cfg.BACKBONE_STRIDES]

    return np.array(returnshape)
