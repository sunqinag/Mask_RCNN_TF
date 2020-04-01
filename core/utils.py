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
    print(boxes[0])  # 因为中心点从0开始。第一个锚点的x1 y1为负数

    # image_path = 'data/img/2007_000032.jpg'
    # img = cv2.imread(image_path,1)
    # target_dir = r'E:\Pycharm_project\mask_rcnn_TF\generate_anchor_image\single_anchor'+str(shape[0])
    # for i,box in enumerate(boxes):
    #     box = np.squeeze(box)
    #     p1=(int(box[0]),int(box[1]))
    #     p2=(int(box[2]),int(box[3]))
    #     img = cv2.rectangle(img,p1,p2,(255,0,0))
    #     if not os.path.exists(target_dir):
    #         os.mkdir(target_dir)
        # cv2.imwrite(target_dir+'\\anchor_'+str(i)+'.jpg',img)
    # cv2.imwrite(r'E:\Pycharm_project\mask_rcnn_TF\generate_anchor_image\anchor_'+str(shape[0])+'.jpg', img)
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
    print("batch_size", batch_size)
    for i in range(batch_size):
        print(inputs, i)
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

def visible(img):
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.show()