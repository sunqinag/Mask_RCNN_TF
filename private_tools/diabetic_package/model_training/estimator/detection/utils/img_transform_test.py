import unittest
import numpy as np
import cv2

from . import img_transform
from. import bbox_op


class ImgTransformTest(unittest.TestCase):
    def test_resize_img_and_bbox(self):
        data = np.load(
            './微血管瘤分块数据/3.3万/micro6x6/bbox/' +
            '000010744_20140318085002_900420_45390_0.npz')
        img = data['img']
        bboxes = data['bbox']
        img_ori = img.copy()
        img_label = img
        for bbox in bboxes:
            img_label = cv2.rectangle(img_label,
                                      (bbox[1], bbox[0]),
                                      (bbox[3], bbox[2]),
                                      (0, 255, 0))
        # cv2.imwrite('origin.jpg', img_ori)
        # cv2.imwrite('origin_label.jpg', img_label)
        bboxes = bbox_op.corner_2_center(bboxes)
        img_resized, bboxes_resized = img_transform.resize_img_and_bbox(
            img_ori, bboxes, np.array((450, 450))
        )

        bboxes_resized = bbox_op.center_2_corner(bboxes_resized)
        img_resized_label = img_resized
        for bbox_resized in bboxes_resized:
            bbox_resized = bbox_resized.astype(np.int)
            img_resized_label = cv2.rectangle(
                img_resized_label,
                (bbox_resized[1], bbox_resized[0]),
                (bbox_resized[3], bbox_resized[2]),
                (0, 255, 0))
        # cv2.imwrite('resized_img.jpg', img_resized_label)

    def test_padding_img_and_bbox(self):
        data = np.load(
            './微血管瘤分块数据/3.3万/micro6x6/bbox/' +
            '000010744_20140318085002_900420_45390_0.npz')
        img = data['img']
        bboxes = data['bbox']
        img_ori = img.copy()
        bboxes = bbox_op.corner_2_center(bboxes)
        img_padding, bboxes_padding = img_transform.padding_img_and_bbox(
            img_ori, bboxes, np.array((50, 50))
        )
        img_padding_label = img_padding
        bboxes_padding = bbox_op.center_2_corner(bboxes_padding)
        for bbox_padding in bboxes_padding:
            bbox_padding = bbox_padding.astype(np.int)
            img_padding_label = cv2.rectangle(
                img_padding_label,
                (bbox_padding[1], bbox_padding[0]),
                (bbox_padding[3], bbox_padding[2]),
                (0, 255, 0)
            )
        # cv2.imwrite('padding_img.jpg', img_padding_label)

    def test_block_img_with_bbox(self):
        data = np.load('./微血管瘤分块数据/3.3万/micro6x6/bbox/' +
                       '000010744_20140318085002_900420_45390_0.npz')
        img = data['img']
        bboxes = data['bbox']
        grids = (6, 6)
        bboxes = bbox_op.corner_2_center(bboxes)
        sub_imgs, sub_img_bboxes = img_transform.block_img_with_bbox(
            img, grids, bboxes
        )
        self.assertEqual(len(sub_imgs), len(sub_img_bboxes))
        k = 0
        for img, bboxes in zip(sub_imgs, sub_img_bboxes):
            # cv2.imwrite('./unittest_result/sub' + str(k) + '.jpg', img)
            for bbox in bboxes:
                bbox = bbox_op.center_2_corner(bbox).astype(np.int)
                img = cv2.rectangle(
                    img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 255, 0)
                )
            # cv2.imwrite('./unittest_result/sub' + str(k) + '_.jpg', img)
            k += 1

    def test_restore_block_img_with_bbox(self):
        data = np.load('./微血管瘤分块数据/3.3万/micro6x6/bbox/' +
                       '000016630_20140916094154_900431_3453_1.npz')
        img = data['img']
        bboxes = data['bbox']
        grids = (6, 6)
        bboxes = bbox_op.corner_2_center(bboxes)
        sub_imgs, sub_img_bboxes_list = img_transform.block_img_with_bbox(
            img, grids, bboxes
        )
        restore_img, restore_img_bboxes = \
            img_transform.restore_block_img_with_bbox(
                sub_imgs, grids, sub_img_bboxes_list
            )
        restore_img_bboxes = bbox_op.center_2_corner(
            np.array(restore_img_bboxes))
        # cv2.imwrite('./unittest_result/restore_img.jpg', restore_img)
        restore_label_img = restore_img
        restore_label_img = self.__draw_bboxes(
            restore_label_img, restore_img_bboxes)
        # cv2.imwrite('./unittest_result/restore_label_img.jpg',
        #             restore_label_img)

    def test_random_selected_patch_ad_bboxes(self):
        data = np.load('./微血管瘤分块数据/3.3万/micro6x6/bbox/' +
                       '000016630_20140916094154_900431_3453_1.npz')
        img = data['img']
        bboxes = data['bbox']
        label_img = self.__draw_bboxes(img, bboxes)
        cv2.imwrite('./unittest_result/origin.jpg', label_img)
        y_scale_range = (0.9, 1)
        x_scale_range = (0.1, 0.8)
        bboxes = bbox_op.corner_2_center(bboxes)
        sub_img, bboxes = img_transform.random_crop(
            img, bboxes, y_scale_range, x_scale_range
        )
        self.assertTrue(
            y_scale_range[0] * img.shape[0] <= sub_img.shape[0] <=
            y_scale_range[1] * img.shape[0])
        self.assertTrue(
            x_scale_range[0] * img.shape[1] <= sub_img.shape[1] <=
            x_scale_range[1] * img.shape[1])
        bboxes = bbox_op.center_2_corner(bboxes)
        result = self.__draw_bboxes(sub_img, bboxes)
        cv2.imwrite('./unittest_result/random_sample_img.jpg', result)

    def test_random_crop_accord_to_bbox(self):
        data = np.load('./微血管瘤分块数据/3.3万/micro6x6/bbox/' +
                       '000016630_20140916094154_900431_3453_1.npz')
        img = data['img']
        bboxes = data['bbox']
        label_img = self.__draw_bboxes(img, bboxes)
        cv2.imwrite('./unittest_result/origin.jpg', label_img)
        y_scale_range = (0.9, 1)
        x_scale_range = (0.1, 0.8)
        bboxes = bbox_op.corner_2_center(bboxes)
        crop_img_list, bboxes_list = img_transform.random_crop_accord_to_bbox(
            img, bboxes, y_scale_range, x_scale_range
        )
        self.assertTrue(len(crop_img_list), len(bboxes_list))
        k = 0
        for img, bboxes in zip(crop_img_list, bboxes_list):
            bboxes = bbox_op.center_2_corner(bboxes)
            result = self.__draw_bboxes(img, bboxes)
            cv2.imwrite('./unittest_result/crop_bbox' + str(k) + '.jpg', result)
            k += 1

    def test_flip_img_and_bboxes(self):
        data = np.load('./微血管瘤分块数据/3.3万/micro6x6/bbox/' +
                       '000016630_20140916094154_900431_3453_1.npz')
        img = data['img']
        bboxes = data['bbox']
        bboxes = bbox_op.corner_2_center(bboxes)
        flip_img, flip_bboxes = img_transform.flip_img_and_bboxes(img, bboxes)
        flip_bboxes = bbox_op.center_2_corner(flip_bboxes)
        flip_label_img = self.__draw_bboxes(flip_img, flip_bboxes)
        cv2.imwrite('./unittest_result/flip.jpg', flip_label_img)

    def test_scale_img_and_bboxes(self):
        data = np.load('./微血管瘤分块数据/3.3万/micro6x6/bbox/' +
                       '000016630_20140916094154_900431_3453_1.npz')
        img = data['img']
        bboxes = data['bbox']
        bboxes = bbox_op.corner_2_center(bboxes)
        scale_img, scale_bboxes = img_transform.scale_img_and_bboxes(
            img, bboxes, (0.5, 1.6))
        scale_bboxes = bbox_op.center_2_corner(scale_bboxes)
        scale_label_img = self.__draw_bboxes(scale_img, scale_bboxes)
        cv2.imwrite('./unittest_result/scale.jpg', scale_label_img)

    def __draw_bboxes(self, img, bboxes):
        label_img = img.copy()
        for bbox in bboxes:
            bbox = bbox.astype(np.int)
            label_img = cv2.rectangle(
                label_img,
                (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 255, 0)
            )
        return label_img


if __name__ == '__main__':
    unittest.main()
