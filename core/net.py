import tensorflow as tf
import numpy as np
from core.cfg import cfg
import core.utils as utils

tf.enable_eager_execution()

def resnet(input_image, train_bn=True):
    # 第一层特征
    x = tf.identity(input=input_image, name='resnet_input_layer')

    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=3, strides=(1, 1), use_bias=True, name='conv1',
                         padding='same')
    x = tf.layers.batch_normalization(inputs=x, training=train_bn, name='bn_conv1')
    x = tf.nn.relu(x)
    C1 = x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=(2, 2))
    # 第2特征层
    x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=3, strides=(1, 1), use_bias=True, name='conv2',
                         padding='same')
    x = tf.layers.batch_normalization(inputs=x, training=train_bn, name='bn_conv2')
    x = tf.nn.relu(x)
    C2 = x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=(2, 2))
    # 第3特征层
    x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=3, strides=(1, 1), use_bias=True, name='conv3',
                         padding='same')
    x = tf.layers.batch_normalization(inputs=x, training=train_bn, name='bn_conv3')
    x = tf.nn.relu(x)
    C3 = x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=(2, 2))
    # 第4特征层
    x = tf.layers.conv2d(inputs=x, filters=1024, kernel_size=3, strides=(1, 1), use_bias=True, name='conv4',
                         padding='same')
    x = tf.layers.batch_normalization(inputs=x, training=train_bn, name='bn_conv4')
    x = tf.nn.relu(x)
    C4 = x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=(2, 2))
    # 第5特征层
    x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=3, strides=(1, 1), use_bias=True, name='conv5',
                         padding='same')
    x = tf.layers.batch_normalization(inputs=x, training=train_bn, name='bn_conv5')
    x = tf.nn.relu(x)
    C5 = x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=(2, 2))
    return C1, C2, C3, C4, C5


def rpn_layer(inpute_feature_map,  # 输入的特征，其w与h所围成面积的个数当作锚点的个数。
              anchors_per_location,  # 每个待计算锚点的网格，需要划分几种形状的矩形
              anchor_stride,
              num):  # 扫描网格的步长
    # 通过一个卷积得到共享特征
    shared = tf.layers.conv2d(inputs=inpute_feature_map, filters=512, kernel_size=3, strides=anchor_stride,
                              padding='same', activation=tf.nn.relu, name='rpn_conv_shared_' + str(num))
    # 第一部分计算锚点的分数（前景和背景） [batch, height, width, anchors per location * 2].
    x = tf.layers.conv2d(inputs=shared, filters=anchors_per_location * 2, kernel_size=1, activation=tf.nn.relu)

    # 将feature_map展开，得到[batch, anchors, 2]。anchors=feature_map的h*w*anchors_per_location
    rpn_class_logits = tf.reshape(x, (tf.shape(x)[0], -1, 2))

    # 用Softmax来分类前景和背景BG/FG.结果当作分数
    rpn_probs = tf.nn.softmax(rpn_class_logits, name='rpn_softmax_XXX_' + str(num))

    # 第二部分计算锚点的边框，每个网格划分anchors_per_location种矩形框，每种4个坐标
    x = tf.layers.conv2d(inputs=shared, filters=anchors_per_location * 4, kernel_size=1, activation=tf.nn.relu,
                         name='rpn_bbox_pred_' + str(num))

    # 将feature_map展开，得到[batch, anchors, 4]
    rpn_bbox = tf.reshape(x, (tf.shape(x)[0], -1, 4),name='rpn_box_reshape')
    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_layer(inpute_feature_map,  # 输入的特征，其w与h所围成面积的个数当作锚点的个数。
                    anchors_per_location,  # 每个待计算锚点的网格，需要划分几种形状的矩形
                    anchor_stride):
    '''
    这里rpn给出的prob_box为dx，dy，log(dw),log(dh)更多的是给出的拟合的优化参量。而不是真实坐标。参照博客https://zhuanlan.zhihu.com/p/31426458
    :param inpute_feature_map:
    :param anchors_per_location:
    :param anchor_stride:
    :return:
    '''
    out2 = rpn_layer(inpute_feature_map[0], anchors_per_location, anchor_stride, num=2)
    out3 = rpn_layer(inpute_feature_map[1], anchors_per_location, anchor_stride, num=3)
    out4 = rpn_layer(inpute_feature_map[2], anchors_per_location, anchor_stride, num=4)
    out5 = rpn_layer(inpute_feature_map[3], anchors_per_location, anchor_stride, num=5)
    out6 = rpn_layer(inpute_feature_map[4], anchors_per_location, anchor_stride, num=6)
    return [out2, out3, out4, out5, out6]


############################################################
#  Proposal Layer
############################################################
# 按照给定的框与偏移量，计算最终的框
def apply_box_deltas_graph(boxes,  # [N, (y1, x1, y2, x2)]
                           deltas):  # [N, (dy, dx, log(dh), log(dw))]
    # 转换成中心点和h，w格式
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # 计算偏移
    # 先做平移
    center_y += height * deltas[:, 0]
    center_x += width * deltas[:, 1]
    # 在做缩放
    height = height * tf.exp(deltas[:, 2])
    width = width * tf.exp(deltas[:, 3])
    # 再转换成左上右下两个点，y1,x1,y2,x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = center_y + 0.5 * height
    x2 = center_x + 0.5 * width
    result = tf.stack([y1, x1, y2, x2], axis=1, name='apply_box_deltas_out')
    return result


def clip_boxes_graph(boxes,  # 计算完的box[N, (y1, x1, y2, x2)]
                     window):  ##y1, x1, y2, x2[0, 0, 1, 1]
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    cliped = tf.concat([y1, x1, y2, x2], axis=1, name='clipped_boxes')
    return cliped


class ProposalLayer:
    def __init__(self, proposal_count, nms_threshold, batch_size, **kwargs):
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.batch_size = batch_size

    def __call__(self, inputs):
        '''
            输入字段input描述
            rpn_probs: [batch, num_anchors, 2] #(bg概率, fg概率)
            rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
            anchors: `[batch, (y1, x1, y2, x2)]
        '''
        # 将前景概率值取出[Batch, num_anchors, 1]
        scores = inputs[0][:, :, 1]
        # 取出位置偏移量[batch, num_anchors, 4]
        deltas = inputs[1]  # 取值范围在0-1量级
        deltas = deltas * np.reshape(cfg.RPN_BBOX_STD_DEV, [1, 1, 4])  # 缩小将近10倍
        # 取出锚点
        anchors = inputs[2]

        # 获得前6000个分值最大的数据
        pre_nms_limit = tf.minimum(6000, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
        # 获取scores中索引为ix的值
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y), self.batch_size)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y), self.batch_size)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda x, y: tf.gather(x, y),
                                            self.batch_size, names=["pre_nms_anchors"])

        # 得出最终的框坐标。[batch, N,4] (y1, x1, y2, x2),将框按照偏移缩放的数据进行计算，
        boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),
                                  self.batch_size,
                                  names=["refined_anchors"])
        # 对出界的box进行剪辑，范围控制在 0.到1 [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes, lambda x: clip_boxes_graph(x, window),
                                  self.batch_size,
                                  names=["refined_anchors_clipped"])

        # Non-max suppression算法
        def NMS(boxes, scores):
            indices = tf.image.non_max_suppression(boxes, scores, self.proposal_count,
                                                   self.nms_threshold, name='rpn_non_max_suppression')  # 计算nms，并获得索引
            proposal = tf.gather(boxes, indices)  # 在boxes中取出indices索引所指的值
            # 如果proposals的个数小于proposal_count，剩下的补0
            padding = tf.maximum(self.proposal_count - tf.shape(proposal)[0], 0)
            proposal = tf.pad(proposal, [(0, padding), (0, 0)])
            return proposal

        proposal = utils.batch_slice([boxes, scores], NMS, self.batch_size)
        return proposal


# 语义分割
def build_fpn_mask_graph(rois,  # 目标实物检测结果，标准坐标[batch, num_rois, (y1, x1, y2, x2)]
                         feature_maps,  # 骨干网之后的fpn特征[P2, P3, P4, P5]
                         pool_size, num_classes, batch_size, train_bn=True):
    """
    返回: Masks [batch, roi_count, height, width, num_classes]
    """
    # ROIAlign 最终统一池化的大小为14
    # Shape: [batch, boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign(batch_size, [pool_size, pool_size],
                        name='roi_align_mask')([rois, feature_maps])

    # conv_layers
    x = tf.layers.conv3d(x, 256, 3, padding='same', name='mrcnn_mask_conv1')
    x = tf.layers.batch_normalization(x, training=train_bn, name='mrcnn_mask_bn1')
    x = tf.nn.relu(x)

    x = tf.layers.conv3d(x, 256, 3, padding='same', name='mrcnn_mask_conv2')
    x = tf.layers.batch_normalization(x, training=train_bn, name='mrcnn_mask_bn2')
    x = tf.nn.relu(x)

    x = tf.layers.conv3d(x, 256, 3, padding='same', name='mrcnn_mask_conv3')
    x = tf.layers.batch_normalization(x, training=train_bn, name='mrcnn_mask_bn3')
    x = tf.nn.relu(x)

    x = tf.layers.conv3d(x, 256, 3, padding='same', name='mrcnn_mask_conv4')
    x = tf.layers.batch_normalization(x, training=train_bn, name='mrcnn_mask_bn4')
    x = tf.nn.relu(x)  # Tensor("Relu_10:0", shape=(?, 14, 14, 256), dtype=float32)

    # 使用反卷积上采样
    x = tf.layers.conv3d_transpose(x, 256, 2, strides=(2, 2,2), activation=tf.nn.relu, name='mrcnn_masj_deconv')

    # 用卷积代替全连接
    x = tf.layers.conv3d(x, num_classes, 1, activation=tf.nn.relu, name='mrcnn_mask')

    return x


def fpn_classifier_graph(rois, feature_maps,
                         pool_size, num_classes, batch_size, train_bn=True,
                         fc_layers_size=1024):
    # ROIAlign层 Shape: [batch, num_boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign(batch_size, [pool_size, pool_size],
                        name="roi_align_classifier")([rois, feature_maps])
    x = tf.squeeze(x,axis=0)
    # 用卷积替代两个1024全连接网络
    x = tf.layers.conv2d(x, fc_layers_size, pool_size, (pool_size, pool_size), padding='same', name='mrcnn_class_conv1')
    x = tf.layers.batch_normalization(x, training=train_bn, name='mrcnn_class_bn1')
    x = tf.nn.relu(x)
    # 1*1卷积，代替第二个全连接
    x = tf.layers.conv2d(x, fc_layers_size, 1, name='mecnn_class_conv2')
    x = tf.layers.batch_normalization(x, training=train_bn, name='mrcnn_class_bn2')
    shared = tf.nn.relu(x)

    # 共享特征层用于计算分类和边框
    mrcnn_class_logits = tf.layers.dense(shared, num_classes, name='mrcnn_class_logots')
    mrcnn_probs = tf.nn.softmax(mrcnn_class_logits, name='mrcnn_probs')

    # 计算边框坐标bbox（偏移量和所放量）
    # [batch, boxes, num_classes * (dy, dx, log(dh), log(dw))]
    x = tf.layers.dense(shared, num_classes * 4, activation=tf.nn.relu)
    mrcnn_bbox = tf.reshape(x, [-1, num_classes, 4], name='mrcnn_box')
    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


class PyramidROIAlign:
    def __init__(self, batch_size, pool_shape, **kwargs):
        self.pool_shape = tuple(pool_shape)
        self.batch_size = batch_size

    def log2_graph(self, x):  # 计算log2
        return tf.log(x) / tf.log(2.0)

    def __call__(self, inputs):
        '''
        输入参数 Inputs:
        -ROIboxes(RPN结果): [batch, num_boxes, 4]，4：(y1, x1, y2, x2)。nms后得锚点坐标.num_boxes=1000
        - image_meta: [batch, (meta data)] 图片的附加信息 93
        - Feature maps: [P2, P3, P4, P5]骨干网经过fpn后的特征.每个[batch, height, width, channels]
        [(1, 256, 256, 256),(1, 128, 128, 256),(1, 64, 64, 256),(1, 32, 32, 256)]
        '''
        # 获取输入参数
        ROIboxes = inputs[0]  # (1, 1000, 4)
        feature_maps = inputs[1]

        # 将锚点坐标提出来
        y1, x1, y2, x2 = tf.split(ROIboxes, 4, axis=2)  # [batch, num_boxes, 4]
        h = y2 - y1
        w = x2 - x1

        ###############################在这1000个ROI里，按固定算法匹配到不同level的特征。
        # 获得图片形状
        image_shape = [cfg.IMAGE_MIN_DIM, cfg.IMAGE_MIN_DIM]
        image_shape = tf.convert_to_tensor(image_shape)
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        # 因为在生成anchor的时候计算公式会保持特征图尺寸与anchor尺寸之间的比例关系恒定，因而可以通过roi面积和特征图面积反推
        # roi所在的层级关系
        # 因为h与w是标准化坐标。其分母已经被除了tf.sqrt(image_area)。
        # 这里再除以tf.sqrt(image_area)分之1，是为了变为像素坐标
        roi_level = self.log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # 每个roi按照自己的区域去对应的特征里截取内容，并resize成指定的7*7大小. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            # equal会返回一个true false的（1，1000），where返回其中为true的索引[[0,1],[0,4],,,[0,200]]
            ix = tf.where(tf.equal(roi_level, level), name="ix")  # (828, 2)

            # 在多维上建立索引取值[?,4](828, 4)
            level_boxes = tf.gather_nd(ROIboxes, ix, name='ix')  # 函数原型,nd的意思是可以收集n dimension的tensor

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)  # (828, )，【0，0，0，0，0，】如果批次为2，就是[000...111]

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # 下面两个值，是ROIboxes中按照不同尺度划分好的索引，对于该尺度特征中的批次索引，不希望有变化。所以停止梯度
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # 结果: [batch * num_boxes, pool_height, pool_width, channels]
            # feature_maps [(1, 256, 256, 256),(1, 128, 128, 256),(1, 64, 64, 256),(1, 32, 32, 256)]
            # box_indices一共level_boxes个。指定level_boxes中的第几个框，作用于feature_maps中的第几个图片
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape, method="bilinear"))

        # 1000个roi都取到了对应的内容，将它们组合起来。( 1000, 7, 7, 256)
        pooled = tf.concat(pooled, axis=0)  # 其中的顺序是按照level来的需要重新排列成原来ROIboxes顺序，因为roi也是通过range依次抽取组合成

        # 重新排列成原来ROIboxes顺序
        box_to_level = tf.concat(box_to_level, axis=0)  # 按照选取level的顺序
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)  # [1000，3] 3([xi] range)
        # 取出头两个批次+序号（1000个），每个值代表原始roi展开的索引了。
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]  # 保证一个批次在100000以内
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(  # 按照索引排序，
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)  # 将roi中的顺序对应到pooled中的索引取出来
        pooled = tf.gather(pooled, ix)  # 按照索引从pooled中取出的框，就是原始顺序了。

        # 加上批次维度，并返回
        pooled = tf.expand_dims(pooled, 0)  # 应该用reshape
        # pooled = KL.Reshape([self.batch_size,-1, self.pool_shape, self.pool_shape, mask_rcnn_model.FPN_FEATURE], name="pooled")(pooled)
        # pooled = tf.reshape(pooled, [self.batch_size,1000, self.pool_shape, self.pool_shape, mask_rcnn_model.FPN_FEATURE])
        return pooled
