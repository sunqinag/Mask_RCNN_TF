import os
import cv2
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from diabetic_package.file_operator import bz_path

IMAGE_SIZE = 512
MAX_OBJ_NUM = 50
ROI_SIZE = 56


class Dataset:
    def __init__(self, mode, base_folder, tfrecord_folder, data_reload=True):
        self.mode = mode
        self.base_folder = base_folder
        img_dir = self.base_folder + os.sep + 'imgs'
        label_dir = self.base_folder + os.sep + 'labels'
        self.img_list = sorted(bz_path.get_file_path(img_dir, ret_full_path=True))
        self.label_list = sorted(bz_path.get_file_path(label_dir, ret_full_path=True))
        self.tfrecord_name = tfrecord_folder + os.sep + str(self.mode) + '.tfrecords'
        if data_reload:
            self.image, self.boxes, self.masks, self.class_ids = self.load_tfrecord(self.tfrecord_name)
        else:
            self.create_tfrecord(self.tfrecord_name)
            self.image, self.boxes, self.masks, self.class_ids = self.load_tfrecord(self.tfrecord_name)

    def get_single_mask(self, src_label):
        '''
        将一个多类别的二值图抓化成多个单类别的二值图，组成一个高维矩阵
        :param src_label:原始二值图label
        :return:
        '''
        image, contours, hierarchy = cv2.findContours(src_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        masks = np.array([])
        bbox = []
        for i in range(0, len(contours)):
            if contours[i].shape[0] >= MAX_OBJ_NUM:
                x, y, w, h = cv2.boundingRect(contours[i])
                # src_img[int(y + h / 2), int(x + w / 2)] = 255
                # print('corrd:', [int(x), int(y), int(x + w / 2), int(y + h / 2)])
                label = src_label[int(y + h / 2), int(x + w / 2)]
                bbox.extend([int(x), int(y), int(x + w / 2), int(y + h / 2), label])

                region = src_label[y:y + h + 1, x:x + w + 1]
                region = cv2.resize(region, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_NEAREST)
                # visible(region)
                mask = np.expand_dims(region, 2)
                masks = np.append(masks, mask)
        if len(bbox) < MAX_OBJ_NUM * 5:
            bbox = bbox + [0, 0, 0, 0, 0] * (MAX_OBJ_NUM - int(len(bbox) / 5))
        else:
            bbox = bbox[:MAX_OBJ_NUM * 5]
        bbox = np.array(bbox).reshape((MAX_OBJ_NUM, 5)).astype(np.float32)
        masks = np.reshape(masks, (ROI_SIZE, ROI_SIZE, -1))
        if masks.shape[-1] < MAX_OBJ_NUM:
            pad_mask = np.tile(np.zeros((ROI_SIZE, ROI_SIZE, 1)), (1, 1, (MAX_OBJ_NUM - masks.shape[-1])))
            masks = np.concatenate([masks, pad_mask], axis=2).astype(np.float32)
        else:
            masks = np.concatenate([masks[:, :, :MAX_OBJ_NUM]], axis=2).astype(np.float32)
        return masks.tostring(), bbox[:, :4].tostring(), bbox[:, 4:].tostring()

    def create_tfrecord(self, tfrecord_name):
        '''
        这里一定注意，存储tfrecord的格式要和解析格式一致，都是float32.全部转成string类型存储比较好，
        至于dataset读取和多线程dataset读取参照博客：https://blog.csdn.net/ricardo232525/article/details/89709793
        :param img_list:
        :param full_scale_mask_list:
        :return:
        '''
        # 定义writer,用于向tfrecord写入数据
        write = tf.python_io.TFRecordWriter(tfrecord_name)
        print('生成tfrecord，存放于：',tfrecord_name)
        for i in tqdm(range(len(self.img_list))):
            img = cv2.imread(self.img_list[i], 1)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            full_scale_mask = cv2.imread(self.label_list[i], 0)
            # print(self.full_scale_mask_list[i])
            masks, bbox, class_ids = self.get_single_mask(full_scale_mask)

            # 将图片转为二进制格式
            img = img.tobytes()

            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                    'masks': tf.train.Feature(bytes_list=tf.train.BytesList(value=[masks])),
                    'bboxes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox])),
                    'class_ids': tf.train.Feature(bytes_list=tf.train.BytesList(value=[class_ids]))
                })
            )
            write.write(example.SerializeToString())
        write.close()
        print('tfrecord制作完毕！！')

    def pareser(self, serialized):
        features = tf.parse_single_example(serialized,  # 取出包含各个feature
                                           features={
                                               'img_raw': tf.FixedLenFeature([], tf.string),
                                               'masks': tf.FixedLenFeature([], tf.string),
                                               'bboxes': tf.FixedLenFeature([], tf.string),
                                               'class_ids': tf.FixedLenFeature([], tf.string)
                                           })
        # 将字符串解析成对应的像素数组
        image = tf.decode_raw(features['img_raw'], tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
        masks = tf.decode_raw(features['masks'], tf.float32)
        print(tf.shape(masks))
        masks = tf.reshape(masks, [ROI_SIZE, ROI_SIZE, MAX_OBJ_NUM])
        boxes = tf.decode_raw(features['bboxes'], tf.float32)
        boxes = tf.reshape(boxes, [MAX_OBJ_NUM, 4])
        class_ids = tf.decode_raw(features['class_ids'], tf.float32)
        class_ids = tf.reshape(class_ids, (MAX_OBJ_NUM,))
        return image, boxes, masks, class_ids

    def load_tfrecord(self, tfrecord_file):
        dataset = tf.data.TFRecordDataset(tfrecord_file)
        dataset = dataset.map(self.pareser)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size=3)
        dataset = dataset.repeat(100)

        iteror = dataset.make_one_shot_iterator()
        image, boxes, masks, class_ids = iteror.get_next()
        return image, boxes, masks, class_ids


if __name__ == '__main__':
    from core.utils import visible

    dataset = Dataset(mode='val', base_folder=r'E:\Pycharm_project\mask_rcnn_TF\voc\val',
                      tfrecord_folder=r'E:\Pycharm_project\mask_rcnn_TF\data',
                      data_reload=False)
    image, boxes, masks, class_ids = dataset.image, dataset.boxes, dataset.masks, dataset.class_ids
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        i = 0
        while True:
            i += 1
            # 顺序获取数据，打印输出
            im, bo, ma, class_id = sess.run([image, boxes, masks, class_ids])
            d=0
            print(i, im, bo, ma)
