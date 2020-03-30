from easydict import EasyDict as edit
import numpy as np

_C =edit()

cfg=_C

#定义全局输入图片大小（二选一）,图片会被下采样6次。必须能够被2的6次方整除
cfg.IMAGE_MIN_DIM = 1024
cfg.IMAGE_MAX_DIM = 1024
#每个锚点的边长初始值
cfg.RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

#在训练过程中，将选取多少个ROI放到fpn层
cfg.TRAIN_ROIS_PER_IMAGE = 32
#BATCH_SIZE大小
cfg.BATCH_SIZE=2

cfg.IMAGE_RESIZE_MODE = "square"#统一成IMAGE_MAX_DIM
# 图片resize时，定义的最小的缩放范围.0代表不进行最小缩放范围限制
cfg.IMAGE_MIN_SCALE = 0
cfg.BACKBONE = "resnet101"     #主干网络使用resnet

#骨干网返回的每一层特征，对原始图片的缩小比例.代表着输出特征的5种尺度
#在计算锚点时，BACKBONE_STRIDES的每个元素，代表按照该像素值划分网格，
#骨干网的输出形状即为256 128 64 32 16,代表输出的网格个数为256 128 64 32 16
cfg.BACKBONE_STRIDES = [4, 8, 16, 32, 64]


#扫描网格的步长。按照该步长获取网格，用于计算锚点。网格中的第一个像素坐标被当作锚点的中心点
cfg.RPN_ANCHOR_STRIDE = 1




#锚点的边长比例(width/height)，将初始值和边长比例一起计算，得到锚点的真实边长。
cfg.RPN_ANCHOR_RATIOS = [0.5, 1, 2]

#训练RPN网络时选取锚点的个数
cfg.RPN_TRAIN_ANCHORS_PER_IMAGE = 256

#训练过程中选取的正向ROI比例
# Percent of positive ROIs used to train classifier/mask heads
cfg.ROI_POSITIVE_RATIO = 0.33

#对应与训练或是使用时，RPN网络最终需要最大保留多少个ROI
cfg.POST_NMS_ROIS_TRAINING = 2000
cfg.POST_NMS_ROIS_INFERENCE = 1000
cfg.RPN_NMS_THRESHOLD = 0.7
cfg.FPN_FEATURE = 256 #特征金字塔层的深度
cfg.DETECTION_MAX_INSTANCES = 100#fpn最终检测的实例个数
#在制作样本的标签时，从一张图中，最多只读取100个实例
cfg.MAX_GT_INSTANCES = 100
#分类时的置信度阈值
cfg.DETECTION_MIN_CONFIDENCE = 0.7
#检测时的Non-maximum suppression阈值
cfg.DETECTION_NMS_THRESHOLD = 0.3

# Pooled ROIs
cfg.POOL_SIZE = 7#金字塔对齐池化后的ROI形状
cfg.MASK_POOL_SIZE = 14
cfg.MASK_SHAPE = [28, 28]


#RPN和最终检测的边界框细化标准偏差 Bounding box refinement standard deviation for RPN and final detections.
cfg.RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
cfg.BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

#是否对掩码进行压缩
cfg.USE_MINI_MASK = True
cfg.MINI_MASK_SHAPE = (56, 56)  # 压缩后的掩码大小(height, width)

cfg.USE_RPN_ROIS=True #是否使用RPN结果（不适用时，是调试场景，手动输入）

cfg.LEARNING_MOMENTUM = 0.9
#梯度剪辑
cfg.GRADIENT_CLIP_NORM = 5.0
# Weight decay regularization
cfg.WEIGHT_DECAY = 0.0001
# Loss weights for more precise optimization.
# Can be used for R-CNN training setup.
cfg.LOSS_WEIGHTS = {
    "rpn_class_loss": 1.,
    "rpn_bbox_loss": 1.,
    "mrcnn_class_loss": 1.,
    "mrcnn_bbox_loss": 1.,
    "mrcnn_mask_loss": 1.
}
# Learning rate and momentum
# The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
# weights to explode. Likely due to differences in optimizer
# implementation.
cfg.LEARNING_RATE = 0.001