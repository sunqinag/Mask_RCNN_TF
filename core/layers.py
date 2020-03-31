import tensorflow as tf


class DetectionTargetLayer():
    def __init__(self, batch_size,name, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.name = name

    def detect(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]



