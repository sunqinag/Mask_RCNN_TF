import numpy as np
import unittest

from . import generate_yolo_dataset


def sigmoid_reverse(x):
    epsilon = 0.0000001
    return np.log(x / (1 - x) + epsilon)


class GenerateYoloDataSetTest(unittest.TestCase):
    def test_microaneurysm_label(self):
        label_func = generate_yolo_dataset.GenerateMicroaneurysmYoloLabel()
        img_hw = (500, 600)
        bboxes = np.array([
            [10, 10,  10, 10],
            [20, 10, 6, 6],
            [100, 305, 60, 60],  # 骑在边界上的情况
            [66, 66, 6, 6],
            [76, 76, 10, 10],
            [80, 80, 1, 1]
        ])
        grids = [10, 10]
        priors = np.array([
            [6, 6],
            [6, 6],
            [6, 6]
        ])
        bboxes_class = np.ones([len(bboxes), 1])
        label_criterion = np.zeros([grids[0], grids[1], len(priors), 7])
        label_criterion[:, :, :, 1] = 1

        label_criterion[0, 0, 0, 0] = 1
        label_criterion[0, 0, 0, -1] = 1
        label_criterion[0, 0, 0, 2: 6] = [
            10 / 50,
            10 / 60,
            np.log(10 / 6),
            np.log(10 / 6)
        ]

        label_criterion[0, 0, 1, 0] = 1
        label_criterion[0, 0, 1, -1] = 1
        label_criterion[0, 0, 1, 2: 6] = [
            20 / 50,
            10 / 60,
            np.log(6 / 6),
            np.log(6 / 6)
        ]

        label_criterion[2, 5, 0, 0] = 1
        label_criterion[2, 5, 0, -1] = 1
        label_criterion[2, 5, 0, 2: 6] = [
            0,
            5 / 60,
            np.log(60 / 6),
            np.log(60 / 6)
        ]

        label_criterion[1, 1, 0, 0] = 1
        label_criterion[1, 1, 0, -1] = 1
        label_criterion[1, 1, 0, 2: 6] = [
            16 / 50,
            6 / 60,
            np.log(6 / 6),
            np.log(6 / 6)
        ]

        label_criterion[1, 1, 1, 0] = 1
        label_criterion[1, 1, 1, -1] = 1
        label_criterion[1, 1, 1, 2: 6] = [
            26 / 50,
            16 / 60,
            np.log(10 / 6),
            np.log(10 / 6)
        ]

        label_criterion[1, 1, 2, 0] = 1
        label_criterion[1, 1, 2, -1] = 1
        label_criterion[1, 1, 2, 2: 6] = [
            30 / 50,
            20 / 60,
            np.log(1 / 6),
            np.log(1 / 6)
        ]

        label = label_func(img_hw, bboxes, bboxes_class, grids, priors)
        epsilon = 0.000001
        diff = np.abs(label - label_criterion)
        inds = np.where(diff > epsilon)
        print(inds)
        print(diff[inds[0], inds[1], inds[2], inds[3]])
        print(label_criterion[inds[0], inds[1], inds[2], inds[3]])
        print(label[inds[0], inds[1], inds[2], inds[3]])
        self.assertTrue((np.abs(label - label_criterion) <= epsilon).all())


if __name__ == '__main__':
    unittest.main()
