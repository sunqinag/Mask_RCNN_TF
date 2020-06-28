'''
        这个脚本用来生成训练YOLO需要的dataset，生成的结果的数据结构是：
    一个ndarray，ndarray的shape为
    [grid_rows, grid_cols, predictor_num_per_cell, 6 + C]，每个prior要的标签
    的维度是5 + C，其中C为object的类别数。
        6 + C维标签的排列顺序为[objectness, objectness_loss_calc, t_y, t_x, t_h,
        t_w, class_label],
    objectness:
        标识prior所在grid cell是否包含object（即包含object的grid cell中与object
        IoU最大的prior这个标签会设置为1，否则设置为0)
    objectness_loss_calc:
        标识这个prior是否参与objectness loss的计算，如果这个prior与ground truth
        的IoU超过一定阈值，但又不是最大的，这个值设置为0，否则设置为1
    t_y:
        bbox中心点y坐标相对于所在grid cell左上角y坐标的相对距离
    t_x:
        bbox中心点x坐标相对于所在grid cell左上角x坐标的相对距离
    t_h:
        按照公式b_h = p_h * exp(t_h)，对ground truth中bbox高度反变换的值
    t_w:
        按照公式b_w = p_w * exp(t_w)，对ground truth中bbox宽度反变换的值
'''
import numpy as np

from ..utils import bbox_op


class GenerateYoloLabel():
    def __call__(self, img_hw, bboxes, bboxes_class, grids, priors):
        '''
        img_hw:
            图像的宽高
        bboxes:
            bounding box的坐标(center_y, center_x, height, width), shape=[
            bbox_num, 4]
        bboxes_class:
            bounding box的类别标签, shape=[bbox_num, obj_class_num]
        grids:
            y和x方向上分的grid的个数
        priors:
            每一个grid cell中的设置的bbox predict prior，shape=[
            priors_num_per_cell, 2]，第二个维度依次是prior的[height, width]，
            height和width分别是prior bbox的宽和高

        返回值：
            每个bbox prior对应的标签，shape=[grid_rows, grid_cols,
            priors_per_cell, 6 + obj_class_num], 6 + obj_class_num维标签的排列
            顺序为[objectness, objectness_loss_calc, t_y, t_x, t_h, t_w,
            class_label]
        '''
        self.__check_param(img_hw, bboxes, grids, bboxes_class)
        self.img_height = img_hw[0]
        self.img_width = img_hw[1]
        self.bboxes = bboxes
        self.bboxes_class = bboxes_class
        self.grids_y = grids[0]
        self.grids_x = grids[1]
        self.grid_height = self.img_height / self.grids_y
        self.grid_width = self.img_width / self.grids_x
        self.priors = np.array(priors)
        self.iou_func = bbox_op.IoU()
        obj_class_num = bboxes_class.shape[-1]
        self.label = np.zeros(
            [grids[0], grids[1], len(priors), 6 + obj_class_num])
        self.label[:, :, :, 1] = 1  # 默认的情况下所有prior都参与obj loss计算
        for bbox, bbox_class in zip(bboxes, bboxes_class):
            grid_ind_y, grid_ind_x = self.__get_bbox_center_grid(bbox)
            obj_prior_ind, obj_not_calc_prior_ind = self._prior_objectness(
                grid_ind_y, grid_ind_x, bbox)
            if obj_prior_ind is not None:
                prior = self.priors[obj_prior_ind]
                bbox_t = bbox_op.b2t_yolo(
                    bbox, grid_ind_y, grid_ind_x, prior[0], prior[1],
                    self.grid_height, self.grid_width)
                self.label[grid_ind_y, grid_ind_x, obj_prior_ind] = \
                    np.concatenate([[1], [1], bbox_t, bbox_class])
            if obj_not_calc_prior_ind is not None:
                self.label[grid_ind_y, grid_ind_x, obj_not_calc_prior_ind] = \
                    np.concatenate([[0], [0], np.ones(4 + len(bbox_class))])
        return self.label

    def __check_param(self, img_hw, bboxes, grids, bboxes_class):
        if len(img_hw) != 2:
            raise ValueError('img_hw必须是2维数组')
        if len(bboxes) != len(bboxes_class):
            raise ValueError('bboxes和bboxes_class的个数必须相同')
        if len(grids) != 2:
            raise ValueError('grid的长度必须是2')

    def __get_bbox_center_grid(self, bbox):
        if bbox[0] >= self.img_height or bbox[1] >= self.img_width:
            raise ValueError('Bbox超出图像范围')
        grid_ind_y = np.floor(
            bbox[0] / self.grid_height).astype(np.int)
        grid_ind_x = np.floor(
            bbox[1] / self.grid_width).astype(np.int)
        return grid_ind_y, grid_ind_x

    def _prior_objectness(self, grid_ind_y, grid_ind_x, obj_bbox):
        '''
            返回一个grid cell中object标识位为1的prior的索引和不计入objectness
            loss计算的prior的索引
        '''
        prior_c_y = (grid_ind_y + 0.5) * self.grid_height
        prior_c_x = (grid_ind_x + 0.5) * self.grid_width
        priors_bbox_up = prior_c_y - self.priors[:, 0] / 2
        priors_bbox_left = prior_c_x - self.priors[:, 1] / 2
        priors_bbox_down = prior_c_y + self.priors[:, 0] / 2
        priors_bbox_right = prior_c_x + self.priors[:, 1] / 2
        priors_bbox = np.concatenate(
            [priors_bbox_up,
             priors_bbox_left,
             priors_bbox_down,
             priors_bbox_right],
            axis=1)
        ious = self.iou_func(priors_bbox, obj_bbox)
        obj_prior_ind = np.argmax(ious)
        obj_not_calc_prior_ind = np.where(np.logical_and(
            ious > 0.5, ious < np.max(ious)))[0]
        if obj_not_calc_prior_ind.shape[0] == 0:
            obj_not_calc_prior_ind = None
        return obj_prior_ind, obj_not_calc_prior_ind


class GenerateMicroaneurysmYoloLabel(GenerateYoloLabel):
    def _prior_objectness(self, grid_ind_y, grid_ind_x, obj_box):
        priors_in_cell = self.label[grid_ind_y, grid_ind_x]
        objectnesses = priors_in_cell[:, 0]
        not_used_prior_inds = np.where(objectnesses == 0)[0]
        if not_used_prior_inds.shape[0] != 0:
            obj_prior_ind = not_used_prior_inds[0]
        else:
            obj_prior_ind = None
        return obj_prior_ind, None
