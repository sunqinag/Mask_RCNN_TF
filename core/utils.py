import numpy as np
import cv2,os
import tensorflow as tf
import matplotlib.pyplot as plt


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    anchors = []
    for i in range(len(scales)):  # 遍历不同的尺度。生成锚点
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)  # [anchor_count, (y1, x1, y2, x2)]

def generate_pyramid_anchors_tf(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    anchors = []
    for i in range(len(scales)):  # 遍历不同的尺度。生成锚点
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return tf.concat(anchors, axis=0)  # [anchor_count, (y1, x1, y2, x2)]


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    shape  骨干网输出的特征 256 128 64 32 16 feature_stride尺度BACKBONE_STRIDES[4, 8, 16, 32, 64]  相乘=1024
    按照BACKBONE_STRIDES个像素为单位，在图片上划分网格。得到的网格按照anchor_stride进行计算是否需要算做锚点。
    anchor_stride=1表明都要被用作计算锚点，2表明隔一个取一个网格用于计算锚点。
    每个网格第一个像素为中心点。
    边长由scales按照ratios种比例计算得到。每个中心点配上每种边长，组成一个锚点。
    """
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()  # 复制了ratios个 scales--【32，32，32】
    ratios = ratios.flatten()  # 因为scales只有1个元素。所以不变

    # 在以边长为scales下，将比例开方。再计算边长，另边框相对不规则一些
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # 计算像素点为单位的网格位移,映射，比如这里feature尺寸为256,256，这是在上面每隔一个去一个坐标点再乘以缩小比放大回去，shift个数不变值变大了，间隔为feature_stride
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)  # 得到了x位移和y的位移，构建了整张原图上的位移标框
    # x 【【0，4，8】          y 【【0，0，0】
    #    【0，4，8】             【4，4，4】
    #    【0，4，8】】            【8，8，8】】

    # 以每个网格第一点当作中心点，按照3种边长，为锚点大  小
    # box_width, box_center_x = np.meshgrid(widths, shifts_x)
    # box_height, box_conter_y = np.meshgrid(heights, shifts_y)
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    # w 【【0.5，1，2】          x 【【0，0，0】
    #    【0.5，1，2】             【4，4，4】
    #    【0.5，1，2】             【8，8，8】
    # w2  【0.5，1，2】          2  【0，0，0】
    #    【0.5，1，2】             【4，4，4】
    #    【0.5，1，2】】            【8，8，8】】
    box_centers = np.stack([box_centers_x, box_centers_y], axis=-1).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=-1).reshape([-1, 2])

    # 将中心点,边长转化为两个点的坐标。 (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    # print(boxes[0])  # 因为中心点从0开始。第一个锚点的x1 y1为负数

    return boxes


def generate_anchors_tf(scales, ratios, shape, feature_stride, anchor_stride):
    """
    shape  骨干网输出的特征 256 128 64 32 16 feature_stride尺度BACKBONE_STRIDES[4, 8, 16, 32, 64]  相乘=1024
    按照BACKBONE_STRIDES个像素为单位，在图片上划分网格。得到的网格按照anchor_stride进行计算是否需要算做锚点。
    anchor_stride=1表明都要被用作计算锚点，2表明隔一个取一个网格用于计算锚点。
    每个网格第一个像素为中心点。
    边长由scales按照ratios种比例计算得到。每个中心点配上每种边长，组成一个锚点。
    """
    scales, ratios = tf.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()  # 复制了ratios个 scales--【32，32，32】
    ratios = ratios.flatten()  # 因为scales只有1个元素。所以不变

    # 在以边长为scales下，将比例开方。再计算边长，另边框相对不规则一些
    heights = scales / tf.sqrt(ratios)
    widths = scales * tf.sqrt(ratios)

    # 计算像素点为单位的网格位移,映射，比如这里feature尺寸为256,256，这是在上面每隔一个去一个坐标点再乘以缩小比放大回去，shift个数不变值变大了，间隔为feature_stride
    shifts_y = tf.range(0, shape[0], anchor_stride) * feature_stride
    shifts_x = tf.range(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = tf.meshgrid(shifts_x, shifts_y)  # 得到了x位移和y的位移，构建了整张原图上的位移标框
    # x 【【0，4，8】          y 【【0，0，0】
    #    【0，4，8】             【4，4，4】
    #    【0，4，8】】            【8，8，8】】

    # 以每个网格第一点当作中心点，按照3种边长，为锚点大  小
    # box_width, box_center_x = np.meshgrid(widths, shifts_x)
    # box_height, box_conter_y = np.meshgrid(heights, shifts_y)
    box_widths, box_centers_x = tf.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = tf.meshgrid(heights, shifts_y)
    # w 【【0.5，1，2】          x 【【0，0，0】
    #    【0.5，1，2】             【4，4，4】
    #    【0.5，1，2】             【8，8，8】
    # w2  【0.5，1，2】          2  【0，0，0】
    #    【0.5，1，2】             【4，4，4】
    #    【0.5，1，2】】            【8，8，8】】
    box_centers = tf.stack([box_centers_x, box_centers_y], axis=-1).reshape([-1, 2])
    box_sizes = tf.stack([box_heights, box_widths], axis=-1).reshape([-1, 2])

    # 将中心点,边长转化为两个点的坐标。 (y1, x1, y2, x2)
    boxes =tf.concat([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    # print(boxes[0])  # 因为中心点从0开始。第一个锚点的x1 y1为负数

    return boxes


def norm_boxes(boxes, shape):
    """
    将像素坐标转化为标注化坐标.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates，取值0-image_size
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates 取值0-1
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
        computation graph and then combines the results. It allows you to run a
        graph on a batch of inputs even if the graph is written to support one
        instance only.

        inputs: list of tensors. All must have the same first dimension length
        graph_fn: A function that returns a TF tensor that's part of a graph.
        batch_size: number of slices to divide the data into.
        names: 如果给出，长度应跟输入对应，每个tensor对应一个name
        """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)

    outputs = list(zip(*outputs))
    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result



############################################################
#  Bounding Boxes
############################################################

def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou