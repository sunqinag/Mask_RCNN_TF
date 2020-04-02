import tensorflow as tf
import numpy as np


def __bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def __float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def __int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_yolo_example(img, label):
    '''
        用于生成写入tfrecord文件的example字符串
    '''
    img = img.astype(np.float32)
    label = label.astype(np.float32)
    feature = {
        'img': __float_feature(img.reshape(-1)),
        'label': __float_feature(label.reshape(-1))
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def parse_yolo_example(example_proto, img_shape, label_shape):
    '''
        将tfrecord文件中存储的example解析成img和yolo模型使用的label
    '''
    feature_description = {
        'img': tf.FixedLenFeature(img_shape, tf.float32),
        'label': tf.FixedLenFeature(label_shape, tf.float32)
    }
    features = tf.parse_single_example(example_proto, feature_description)
    img_tensor = features['img']
    label_tensor = features['label']
    return {'img': img_tensor}, label_tensor
