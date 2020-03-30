import tensorflow as tf


class YoloLoss:
    def __init__(
        self, grids, coord_weight, obj_weight, noobj_weight, cls_weight
    ):
        self.grids = grids
        self.coord_weight = coord_weight
        self.obj_weight = obj_weight
        self.noobj_weight = noobj_weight
        self.cls_weight = cls_weight

    def __call__(self,
                 bboxes_logit,
                 objectness_logits,
                 class_logits,
                 bboxes_label,
                 objectnesses_label,
                 classes_label,
                 obj_loss_calc):
        '''
        bboxes_logits:
            bbox的预测结果(Yolo损失函数定义的空间下的坐标)，shape=[batch_size,
            grid_y_num, grid_x_num, predictor_num_per_cell, 4]
        objectness_pre:
            predictor包含object的预测概率，shape=[batch_size, grid_y_num,
            grid_x_num, predictor_num_per_cell]
        class_probs_pre:
            predictor对object类别预测的概率，shape=[batch_size, grid_y_num,
            grid_x_num, predictor_num_per_cell, class_num]
        bboxes_label:
            predictor对应ground truth中的bbox，shape=[batch_size, grid_y_num,
            grid_x_num, predictor_num_per_cell, 4]，最后四维依次是
            [center_y, center_x, height, width](Yolo损失函数定义的空间下的坐标)
        objectnesses_label:
            标识当前predictor所在的grid cell是否包含object，shape=[batch_size,
            grid_y_num, grid_x_num, predictor_num_per_cell]
        classes_label:
            predictor所在grid cell包含的object的类别，shape=[batch_size,
            grid_y_num, grid_x_num, predictor_num_per_cell, class_num]
        obj_loss_calc:
            标识当前predictor是否参与objectness loss的计算，shape=
            [batch_size, grid_y_num, grid_x_num, predictor_num_per_cell]

        返回值:
            不同的类别loss的加权和
        '''
        self.__bboxes_logit = bboxes_logit
        self.__objectnesses_logits = objectness_logits
        self.__class_logits = class_logits
        self.__bboxes_label = bboxes_label
        self.__objectnesses_label = objectnesses_label
        self.__classes_label = classes_label
        self.__obj_loss_calc = obj_loss_calc
        coord_loss = self.__coord_loss()
        obj_loss = self.__objectness_loss()
        classify_loss = self.__class_prob_loss()
        total_loss = coord_loss + obj_loss + classify_loss
        return {'loss': total_loss,
                'coord_loss': coord_loss,
                'obj_loss': obj_loss,
                'classify_loss': classify_loss}

    def __coord_loss(self):
        with tf.name_scope('coord_loss'):
            center_losses = (
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.__bboxes_label[:, :, :, :, 0],
                    logits=self.__bboxes_logit[:, :, :, :, 0]
                ) + tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.__bboxes_label[:, :, :, :, 1],
                    logits=self.__bboxes_logit[:, :, :, :, 1]
                )
            ) * self.__objectnesses_label
            width_height_losses = (
                tf.squared_difference(self.__bboxes_logit[:, :, :, :, 2],
                                      self.__bboxes_label[:, :, :, :, 2]) +
                tf.squared_difference(self.__bboxes_logit[:, :, :, :, 3],
                                      self.__bboxes_label[:, :, :, :, 3])
            ) * self.__objectnesses_label
            center_loss = tf.reduce_sum(center_losses) / tf.reduce_sum(
                self.__objectnesses_label)
            width_height_loss = tf.reduce_sum(width_height_losses) / \
                tf.reduce_sum(self.__objectnesses_label)
            coord_loss = (center_loss + width_height_loss) * self.coord_weight
            tf.add_to_collection(tf.GraphKeys.LOSSES, coord_loss)
            tf.summary.scalar('coord_loss', coord_loss)
            tf.summary.scalar('location_loss', center_loss)
            tf.summary.scalar('size_loss', width_height_loss)
            return coord_loss

    def __objectness_loss(self):
        with tf.name_scope('objectness_loss'):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.__objectnesses_label,
                logits=self.__objectnesses_logits)
            weight_tensor = self.__objectnesses_label * self.obj_weight + \
                (1 - self.__objectnesses_label) * self.noobj_weight
            losses = losses * weight_tensor
            objectness_loss = tf.reduce_sum(
                losses * self.__obj_loss_calc) / tf.reduce_sum(
                    self.__obj_loss_calc)
            tf.add_to_collection(tf.GraphKeys.LOSSES, objectness_loss)
            tf.summary.scalar('obj_loss', objectness_loss)
            return objectness_loss

    def __class_prob_loss(self):
        with tf.name_scope('classify_loss'):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.__classes_label, logits=self.__class_logits)
            class_prob_loss = tf.reduce_sum(
                losses * self.__objectnesses_label[:, :, :, :, tf.newaxis]) /\
                tf.reduce_sum(self.__objectnesses_label)
            tf.add_to_collection(tf.GraphKeys.LOSSES, class_prob_loss)
            tf.summary.scalar('classify_loss', class_prob_loss)
            return class_prob_loss
