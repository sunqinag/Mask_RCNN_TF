import tensorflow as tf
import core.net as net
from core.cfg import cfg
import core.utils as utils
import math
import numpy as np
from core.layers import DetectionTargetLayer, overlaps_graph


class Mask_RCNN:
    def __init__(self, mode, input_rpn_match, input_rpn_bbox):
        """
        mode: 可以是 "training" 或 "inference" 两种模式
        model_dir: 保存模型的路径
        """
        self.mode = mode
        self.input_rpn_match = input_rpn_match
        self.input_rpn_bbox = input_rpn_bbox
        self.batch_size = cfg.BATCH_SIZE
        self.num_class = 21

    def build(self, input_image, input_gt_class_ids, input_gt_box, input_gt_masks):  # 构建Mask R-CNN架构
        input_image = tf.identity(input_image, name='input_image_layer')
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

            # import cv2, os
            # def save_anchor(image,rpn_rois):
            #     img_path = r'E:\Pycharm_project\Mask_RCNN\data\img\2007_000032.jpg'
            #     img = cv2.imread(img_path, 1)
            #     for i in range(rpn_rois.shape[1]):
            #         box = rpn_rois[0, i, :]
            #         p1 = (int(box[0]), int(box[1]))
            #         p2 = (int(box[2]), int(box[3]))
            #         im = cv2.rectangle(img, p1, p2, (0, 0, 255))
            #         print('saving ' + r'E:\Pycharm_project\Mask_RCNN\anchor' + os.sep + str(i) + '.jpg')
            #         cv2.imwrite(r'E:\Pycharm_project\Mask_RCNN\anchor' + os.sep + str(i) + '.jpg', im)
            #     return im
            # val = tf.py_func(save_anchor, [input_image,rpn_rois], [tf.int32])

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
            for i in range(self.batch_size):
                self.build_rpn_targets(anchors=anchors[i, :, :], gt_class_ids=input_gt_class_ids[i, :],
                                       gt_boxes=input_gt_box[i, :, :])
            d = 0

    def build_rpn_targets(self, anchors, gt_class_ids, gt_boxes):
        '''
        给定锚点和GT框，计算重叠并确定正值
        锚点和三角洲以优化它们以匹配其相应的GT_box

        anchors: [num_anchors, (y1, x1, y2, x2)]
        gt_class_ids: [num_gt_boxes] Integer class IDs.
        gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

        Returns:
        rpn_match: [N] (int32) matches between anchors and GT boxes.
                   1 = positive anchor, -1 = negative anchor, 0 = neutral（iou在0.3和0.7之间）
        rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
        '''

        # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
        rpn_match = tf.Variable(tf.zeros([anchors.shape[0]]))#这里经过了修改否则rpn_match是常亮后面无法更改了

        # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
        # 其实rpn_bbox可以砍掉一般。因为只有放了一半的正锚点
        rpn_box = tf.zeros([cfg.RPN_TRAIN_ANCHORS_PER_IMAGE, 4])

        # A crowd box is given a negative class ID.
        crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
        if crowd_ix.shape[0] > 0:
            # Filter out crowds from ground truth class IDs and boxes
            non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
            crowd_box = tf.gather(gt_boxes, crowd_ix)

            gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
            gt_boxes = tf.gather(gt_boxes, non_crowd_ix)

            # Compute overlaps with crowd boxes [anchors, crowds]
            crowd_overlaps = overlaps_graph(anchors, crowd_box)
            crowd_iou_max = tf.argmax(crowd_overlaps)
            non_corwd_bool = (crowd_iou_max < 0.001)
        else:
            # All anchors don't intersect a crowd
            non_crowd_bool = tf.ones([anchors.shape[0]], dtype=tf.bool)

        # Compute overlaps [num_anchors, num_gt_boxes]  每个值都是面积重叠的比例
        overlaps = overlaps_graph(anchors, gt_boxes)
        # 将锚点匹配到GT Boxes
        # 如果锚与IoU> = 0.7的GT框重叠，则为正。
        # 如果锚点与IoU <0.3的GT框重叠，则为负。
        # 中性锚是指不符合上述条件的锚，
        # 而且它们不会影响损失函数。
        # 但是，不要让任何GT_box都无匹配项（稀有，但是会发生）。
        # 相应的，
        # 使它与最接近的锚点匹配（即使其最大IoU <0.3）。

        # 1.首先设置负样本anchor。 如果GT_box是与他们匹配。 跳过corwd_box。
        anchor_iou_argmax = tf.argmax(overlaps, axis=1)
        rpn_match[
            (tf.cast(anchor_iou_argmax, tf.float32) < 0.3) & (non_crowd_bool)]=-1 # 将Iou《0.3的设为-1,同时要保证gt的label值要大于0



        # 为每一个gt_box设置一个锚点，无论IoU的值如何，
        # If multiple anchors have the same IoU match all of them
        gt_iou_argmax = tf.argmax(overlaps, axis=0)
        rpn_match[gt_iou_argmax] = 1  # 有n个gt_box,这必定对应4个与他们iou值最大的anchor，将这些anchor设置为1
        # 3.将重叠度较高的锚定为正。
        rpn_match[tf.cast(anchor_iou_argmax, tf.float32) >= 0.7] = 1  # 将Iou>= 0.7的设为

        # 要么充分重叠，要么不充分重叠。介于二者之间都为0
        # 总共256个正负锚点框，正负锚点超过半数的随机将其去掉
        ids = tf.where(rpn_match == 1)[:, 0]
        extra = tf.cast(ids, tf.int32) - (cfg.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
        if extra > 0:
            # 将多余的重置为中性
            choice = tf.range(ids.shape[0])

            ids == 0
        d = 0

    def get_anchors(self, image_shape):
        '''根据指定图片大小生成锚点.输入为原图尺寸'''
        backbone_shapes = self.compute_backbone_shapes(image_shape)
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

    def compute_backbone_shapes(self, image_shape):
        '''计算主干网络返回的形状'''
        return_shape = [[int(math.ceil(image_shape[0] / stride)), int(math.ceil(image_shape[1] / stride))] for stride in
                        cfg.BACKBONE_STRIDES]
        return np.array(return_shape)
