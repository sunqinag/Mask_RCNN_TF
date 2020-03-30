import unittest
import numpy as np
import tensorflow as tf

import tfexample_converter


class TestTFExampleConverter(unittest.TestCase):
    def test_serialize_and_parse_yolo_example(self):
        def serialize_and_parse(img_shape,
                                grids_y,
                                grids_x,
                                prior_num_per_cell,
                                class_num):
            img = np.random.randint(0, 255, img_shape)
            label = np.random.randint(
                0, 3, [grids_y, grids_x, prior_num_per_cell, 6 + class_num])
            serialized_str = tfexample_converter.serialize_yolo_example(
                img, label)
            features = tfexample_converter.parse_yolo_example(
                serialized_str,
                img_shape,
                label.shape
            )
            with tf.Session() as sess:
                features, label = sess.run(features)
                img_result = features['img']
                label_result = label
            self.assertTrue((img == img_result).all())
            self.assertTrue((label == label_result).all())
        serialize_and_parse((6, 6, 3), 10, 10, 3, 6)
        serialize_and_parse((3, 2), 10, 10, 2, 6)


if __name__ == '__main__':
    unittest.main()
