import cv2, os
import numpy as np
import matplotlib.pyplot as plt



# def get_single_mask(src_label):
#     '''
#     将一个多类别的二值图抓化成多个单类别的二值图，组成一个高维矩阵
#     :param src_label:原始二值图label
#     :return:
#     '''
#     masks = []
#     label = np.unique(src_label)
#     for i in label[1:]:
#         single_mask = np.where(src_label == i, i, 0)
#         masks.append(single_mask)
#     single_mask = np.concatenate([masks], -1).transpose([1, 2, 0])
#     return single_mask

def get_single_mask(src_label):
    '''
    将一个多类别的二值图抓化成多个单类别的二值图，组成一个高维矩阵
    :param src_label:原始二值图label
    :return:
    '''
    image, contours, hierarchy = cv2.findContours(src_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    masks = np.array([])
    bbox = []
    for i in range(0, len(contours)):
        if contours[i].shape[0] >= 50:
            x, y, w, h = cv2.boundingRect(contours[i])
            # src_img[int(y + h / 2), int(x + w / 2)] = 255
            # print('corrd:', [int(x), int(y), int(x + w / 2), int(y + h / 2)])
            label = src_label[int(y + h / 2), int(x + w / 2)]
            bbox.extend([int(x), int(y), int(x + w / 2), int(y + h / 2), label])

            region = src_label[y:y + h + 1, x:x + w + 1]
            region = cv2.resize(region, (56, 56), interpolation=cv2.INTER_NEAREST)
            # visible(region)
            mask = np.expand_dims(region, 2)
            masks = np.append(masks, mask)
    if len(bbox) < 50 * 5:
        bbox = bbox + [0, 0, 0, 0, 0] * (50 - int(len(bbox) / 5))
    else:
        bbox = bbox[:50*5]
    bbox = np.array(bbox).reshape((50, 5)).astype(np.int32)
    masks = np.reshape(masks, (56, 56, -1))
    if masks.shape[-1] < 50:
        pad_mask = np.tile(np.zeros((56, 56, 1)), (1, 1, (50 - masks.shape[-1])))
        masks = np.concatenate([masks, pad_mask], axis=2).astype(np.int32)
    return masks.flatten().tostring(), bbox.flatten().tostring()


def create_tfrecord(img_list, full_scale_mask_list,tfrecord_name):
    '''
    这里一定注意，存储tfrecord的格式要和解析格式一致，都是int32.全部转成string类型存储比较好，
    至于dataset读取和多线程dataset读取参照博客：https://blog.csdn.net/ricardo232525/article/details/89709793
    :param img_list:
    :param full_scale_mask_list:
    :return:
    '''
    # 定义writer,用于向tfrecord写入数据
    write = tf.python_io.TFRecordWriter(tfrecord_name+'.tfrecords')

    for i in tqdm(range(len(img_list))):
        img = cv2.imread(img_list[i], 1)
        img = cv2.resize(img, (500, 500))
        full_scale_mask = cv2.imread(full_scale_mask_list[i], 0)
        print(full_scale_mask_list[i])
        masks, bbox = get_single_mask(full_scale_mask)

        # 将图片转为二进制格式
        img = img.tobytes()

        example = tf.train.Example(
            features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                'masks': tf.train.Feature(bytes_list=tf.train.BytesList(value=[masks])),
                'bboxes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox]))
            })
        )
        write.write(example.SerializeToString())
    write.close()
    print('tfrecord制作完毕！！')


def pareser(serialized):
    features = tf.parse_single_example(serialized,  # 取出包含各个feature
                                       features={
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'masks': tf.FixedLenFeature([], tf.string),
                                           'bboxes': tf.FixedLenFeature([], tf.string)
                                       })
    # 将字符串解析成对应的像素数组
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [500, 500, 3])
    masks = tf.decode_raw(features['masks'], tf.int32)
    print(tf.shape(masks))
    masks = tf.reshape(masks, [56, 56, 50])
    boxes = tf.decode_raw(features['bboxes'], tf.int32)
    boxes = tf.reshape(boxes, [50, 5])
    return image, boxes, masks


if __name__ == '__main__':
    from tqdm import tqdm
    import tensorflow as tf
    from private_tools.file_opter import get_file_path

    img_dir = r'E:\Pycharm_project\mask_rcnn_TF\voc\train\imgs'
    label_dir = r'E:\Pycharm_project\mask_rcnn_TF\voc\train\labels'
    img_list = sorted(get_file_path(img_dir, ret_full_path=True))
    full_scale_mask_list = sorted(get_file_path(label_dir, ret_full_path=True))
    # 制作tfrecord
    create_tfrecord(img_list=img_list, full_scale_mask_list=full_scale_mask_list,tfrecord_name='train')


    # tfrecord_file = r'E:\Pycharm_project\mask_rcnn_TF\mydata.tfrecords'
    #
    # dataset = tf.data.TFRecordDataset(tfrecord_file)
    # dataset = dataset.map(pareser)
    # dataset = dataset.shuffle(buffer_size=1000)
    # dataset = dataset.batch(batch_size=3)
    # dataset = dataset.repeat(100)
    #
    # iteror = dataset.make_one_shot_iterator()
    # image, boxes, masks = iteror.get_next()
    # print(image.shape, boxes.shape, masks.shape)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     i = 0
    #     while True:
    #         i += 1
    #         # 顺序获取数据，打印输出
    #         im, bo, ma = sess.run([image, boxes, masks])
    #         d = 0
    #         print(i, im, bo, ma)
