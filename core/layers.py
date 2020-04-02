import tensorflow as tf
from core.utils import batch_slice
from core.cfg import cfg


class DetectionTargetLayer():
    def __init__(self, batch_size, name, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.name = name

    def __call__(self, inputs, **kwargs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # 按照批次内部的图片，依次调用detection_targets_graph进行最终结果过滤
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(w, x, y, z),
            self.batch_size, names=names)
        return outputs


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks):
    """Generates detection targets for one image. Subsamples proposals and
        generates target class IDs, bounding box deltas, and masks for each.

        Inputs:
        所有gt值输入进来都应该是padding到一个上限的
        proposals: [N, (y1, x1, y2, x2)] in normalized coordinates. Might
                   be zero padded if there are not enough proposals. 取值0-1
        gt_class_ids: [MAX_GT_INSTANCES] int class IDs
        gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
        gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

        Returns: Target ROIs and corresponding class IDs, bounding box shifts,
        and masks.
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
        deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
                Class-specific bbox refinements.
        masks: [TRAIN_ROIS_PER_IMAGE, height, width). Masks cropped to bbox
               boundaries and resized to neural network output size.

        Note: Returned arrays might be zero padded if not enough target ROIs.
        """
    # remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")  # 2000*4
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name='trim_gt_boxes')  # 4*4
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros, name="trim_gt_class_ids")
    # gt_mask_1 = tf.boolean_mask(gt_mask,non_zeros,axis=2,name="trim_gt_masks") #与下方比较两种效果相同
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")  # 这里要转化为下标才好

    # 寻找未激活的类别索引下标
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    # 寻找激活的ids,bbox,mask索引下标
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]

    # 寻找gt中的负样本
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    crowd_masks = tf.gather(gt_masks, crowd_ix, axis=2)

    # 寻找gt中的正样本
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # 计算proposals和gt_boxes重叠率
    overlaps = overlaps_graph(proposals, gt_boxes)

    # 计算anchor与负样本rowd_boxes的重叠率
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)  # 只要未激活的重叠率不超过0.001都算激活

    # 确定激活和未激活的ROIS
    roi_iou_max = tf.reduce_max(overlaps, axis=1)  # 这里reduce_max更多的作用是降维从（2000，？）变成（2000，）
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs,随机选取正样本的33%
    positive_count = int(cfg.TRAIN_ROIS_PER_IMAGE * cfg.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]

    # 负样本，添加足够的量以维持正负比率。
    r = 1.0 / cfg.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]

    # 选择正负ROIS
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # 为gt_box分配正的rois
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= cfg.BBOX_STD_DEV

    # Assign positive ROIs to GT masks
    # Permute masks to [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)
    # Compute mask targets
    boxes = positive_rois
    if cfg.USE_MINI_MASK:
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes, box_ids, cfg.MASK_SHAPE)
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)

    #将阈值蒙版像素设置为0.5可使GT蒙版为0或1，以使用二进制交叉熵损失。
    masks = tf.round(masks)#默认阈值0.5，二值化操作

    # Append negative ROIs and pad bbox deltas and masks that are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(cfg.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks



def box_refinement_graph(box, gt_box):
    """
    Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result



def overlaps_graph(boxes1, boxes2):
    '''
    计算两组boxes之间的iou重叠
    :param boxes1: [N, (y1, x1, y2, x2)].
    :param boxes2: [N, (y1, x1, y2, x2)].
    :return:
    '''
    # 平铺boxes2和重复boxes1。 这使我们可以进行比较每个boxes1对每个boxes2都没有循环。
    # tf没有给出np.repeat等接口，因此用tf.title和tf.reshape等效代替
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])

    # 寻找交集区域坐标并计算面积
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)

    # 计算b1，b2自己的面积，和它们的并集的面积
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection

    # Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps

    # import matplotlib.pyplot as plt
    # for i in range(gt_masks.shape[-1]):
    #     plt.figure(figsize=(6,6))
    #     plt.subplot(121)
    #     plt.title('used non_zeros')
    #     plt.imshow(gt_mask_1[:,:,i])
    #     plt.subplot(122)
    #     plt.title('used tf.gather')
    #     plt.imshow(gt_masks[:,:,i])
    # plt.show()



def trim_zeros_graph(boxes, name=None):
    '''通常，框用形状为[N，4]的矩阵表示用零填充。 这将删除零框。
    boxes: [N, 4] matrix of boxes.
    non_zeros: [N]一维布尔掩码，标识要保留的行
    '''
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1),
                        tf.bool)  # 将四个坐标值累加，变成一个标量，代表某一个box的总值，这个总值是否为0就代表这个box是否是padding来的
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)

    return boxes, non_zeros
