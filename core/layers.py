import tensorflow as tf


class DetectionTargetLayer():
    def __init__(self,batch_size):
        self.batch_size =batch_size
    def call(self, inputs):

