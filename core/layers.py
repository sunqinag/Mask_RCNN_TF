import tensorflow as tf
from core.utils import batch_slice


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


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_mask):
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
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals") #2000*4
    gt_boxes,non_zeros = trim_zeros_graph(gt_boxes,name='trim_gt_boxes') #4*4
    gt_class_ids =tf.boolean_mask(gt_class_ids,non_zeros,name="trim_gt_class_ids")
    gt_mask_1 = tf.boolean_mask(gt_mask,non_zeros,axis=2,name="trim_gt_masks")
    gt_masks = tf.gather(gt_mask, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")#这里要转化为下标才好
    if gt_mask_1 == gt_masks:
        d=1
    else:
        d=2

    d=0


def trim_zeros_graph(boxes, name=None):
    '''通常，框用形状为[N，4]的矩阵表示用零填充。 这将删除零框。
    boxes: [N, 4] matrix of boxes.
    non_zeros: [N]一维布尔掩码，标识要保留的行
    '''
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)  # 将四个坐标值累加，变成一个标量，代表某一个box的总值，这个总值是否为0就代表这个box是否是padding来的
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)

    return boxes, non_zeros
