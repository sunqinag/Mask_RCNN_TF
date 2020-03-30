import unittest
import numpy as np

import bbox_op


def float_equal(a, b):
    return np.abs(a - b) < 0.0000001


class IoUTest(unittest.TestCase):
    def test_exception(self):
        pass

    def test_corner_intersect(self):
        bbox1 = np.array([0, 0, 2, 2])
        bbox2 = np.array([1, 1, 6, 6])
        iou_func = bbox_op.IoU()
        self.assertEqual(iou_func(bbox1, bbox2), 1 / 28)
        self.assertEqual(iou_func(bbox2, bbox1), 1 / 28)

    def test_side_intersect(self):
        bbox1 = np.array([0, 0, 6, 6])
        bbox2 = np.array([1, 1, 3, 8])
        self.assertEqual(bbox_op.IoU()(bbox1, bbox2), 1 / 4)

    def test_in(self):
        bbox1 = np.array([1, 1, 2, 2])
        bbox2 = np.array([-1, -1, 6, 6])
        self.assertEqual(bbox_op.IoU()(bbox1, bbox2), 1 / 49)

    def test_same(self):
        bbox1 = np.array([0, 0, 2, 2])
        self.assertEqual(bbox_op.IoU()(bbox1, bbox1), 1)

    def test_adjacent(self):
        iou_func = bbox_op.IoU()
        bbox1 = np.array([1, 2, 3, 6])
        bbox2 = np.array([3, 6, 5, 8])
        self.assertEqual(iou_func(bbox1, bbox2), 0)
        bbox2 = np.array([5, 11, 20, 20])
        self.assertEqual(iou_func(bbox1, bbox2), 0)

    def test_no_intersection(self):
        bbox1 = np.array([0, 0, 1, 1])
        bbox2 = np.array([2, 2, 3, 3])
        self.assertEqual(bbox_op.IoU()(bbox1, bbox2), 0)

    def test_point_intersect(self):
        bbox1 = np.array([0, 0, 0, 0])
        bbox2 = np.array([1, 1, 1, 1])
        self.assertEqual(bbox_op.IoU()(bbox1, bbox2), 0)

    def test_point_intersect_bbox(self):
        bbox1 = np.array([0, 0, 0, 0])
        bbox2 = np.array([1, 1, 2, 2])
        self.assertEqual(bbox_op.IoU()(bbox1, bbox2), 0)

    def test_output_shape(self):
        iou_func = bbox_op.IoU()
        bbox_single = np.array([3, 3, 6, 6])
        bbox_0d = np.arange(4)
        self.assertEqual(iou_func(bbox_0d, bbox_single), 0)
        bbox_1d = np.array([[3, 3, 5, 5]])
        self.assertTrue((iou_func(bbox_1d, bbox_single) == [4 / 9]).all())
        bbox_2d = np.arange(36).reshape([3, 3, 4])
        self.assertTrue((iou_func(bbox_2d, bbox_single) ==
                         np.array([[0, 2 / 11, 0],
                                   [0, 0, 0],
                                   [0, 0, 0]])).all())


class BboxConvertTest(unittest.TestCase):
    def test_corner_2_center(self):
        bbox_corner = np.array([1, 2, 3, 4])
        bbox_center = bbox_op.corner_2_center(bbox_corner)
        self.assertTrue(float_equal(bbox_center, np.array([2, 3, 2, 2])).all())
        bbox_corner = np.array([[0, 0, 0, 0],
                                [1, 1, 1, 1]])
        bbox_center = bbox_op.corner_2_center(bbox_corner)
        self.assertTrue(float_equal(bbox_center, np.array([
            [0, 0, 0, 0], [1, 1, 0, 0]
        ])).all())
        bbox_corner = np.array([
            [[-1, 2, 3, 2], [1, 2, 1, 2]],
            [[6, 6, 6, 6.6], [6.5, 12.1, 7, 19]]
        ])
        bbox_center = bbox_op.corner_2_center(bbox_corner)
        self.assertTrue(float_equal(bbox_center, np.array([
            [[1, 2, 4, 0], [1, 2, 0, 0]],
            [[6, 6.3, 0, 0.6], [6.75, 15.55, 0.5, 6.9]]
        ])).all())

    def test_center_2_corner(self):
        bbox_center = np.array([0, 0, 2, 1])
        bbox_corner = bbox_op.center_2_corner(bbox_center)
        self.assertTrue(float_equal(bbox_corner,
                                    np.array([-1, -0.5, 1, 0.5])).all())
        bbox_center = np.array([[0, 0, 0, 0], [1, 1, 0, 0]])
        bbox_corner = bbox_op.center_2_corner(bbox_center)
        self.assertTrue(float_equal(bbox_corner, np.array([
            [0, 0, 0, 0], [1, 1, 1, 1]
        ])).all())
        bbox_center = np.array([
            [[1, 2, 3, 4], [5, 6, 0, 8]],
            [[0.5, 0.6, 3, 6], [0.6, 0.6, 0.6, 0.6]]
        ])
        bbox_corner = bbox_op.center_2_corner(bbox_center)
        self.assertTrue(float_equal(bbox_corner, np.array([
            [[-0.5, 0, 2.5, 4], [5, 2, 5, 10]],
            [[-1, -2.4, 2, 3.6], [0.3, 0.3, 0.9, 0.9]]
        ])).all())

    def test_b2t_yolo(self):
        def sigmoid_reverse(y):
            epsilon = 0.000001
            return np.log((y + epsilon) / (1 - y))

        c_y = 6
        c_x = 6
        grid_height = 10
        grid_width = 10
        prior_height = 5
        prior_width = 2
        b = np.array([[61, 62, 63, 64], [66, 68, 69, 69]])
        t = bbox_op.b2t_yolo(
            b, c_y, c_x, prior_height, prior_width, grid_height, grid_width)
        t[:, 0] = sigmoid_reverse(t[:, 0])
        t[:, 1] = sigmoid_reverse(t[:, 1])
        b_ = bbox_op.t2b_yolo(
            t, c_y, c_x, prior_height, prior_width, grid_height, grid_width)
        epsilon = 0.0001
        self.assertTrue((np.abs(b - b_) < epsilon).all())

    def test_t2b_yolo(self):
        pass

    def test_is_bboxes_center_in_grid(self):
        grid = [10, 6, 20, 30]
        bboxes = np.array([
            [11, 8, 6, 6],
            [10, 6, 16, 16],
            [0, 0, 6, 6],
            [20, 29, 6, 6],
            [15, 60, 6, 6],
            [15, 29.6, 6, 6]
        ])
        results = bbox_op.is_bboxes_center_in_grid(bboxes, grid)
        self.assertTrue(
            (results == [True, True, False, False, False, False]).all())


if __name__ == '__main__':
    unittest.main()
