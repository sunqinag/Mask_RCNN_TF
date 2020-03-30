import cv2, os
import numpy as np
import matplotlib.pyplot as plt

def visible(img):
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.show()

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
    X, Y = src_label.shape[0], src_label.shape[1]
    image, contours, hierarchy = cv2.findContours(src_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    masks =np.array([])
    bbox = []
    for i in range(0, len(contours)):
        if contours[i].shape[0] >= 50:
            x, y, w, h = cv2.boundingRect(contours[i])
            # src_img[int(y + h / 2), int(x + w / 2)] = 255
            print('corrd:', [int(x), int(y), int(x + w / 2), int(y + h / 2)])
            label = src_label[int(y + h / 2), int(x + w / 2)].astype(np.float64)
            bbox.extend([int(x), int(y), int(x + w / 2), int(y + h / 2), label])


            region = src_label[y:y + h + 1, x:x + w + 1]
            region = cv2.resize(region,(56,56),interpolation=cv2.INTER_NEAREST)
            # visible(region)
            mask= np.expand_dims(region,2)
            print(np.unique(mask))

            masks = np.append(masks,mask)
    if len(bbox) < 30 * 5:
        bbox = bbox + [0, 0, 0, 0, 0] * (30 - int(len(bbox)/5))
    masks = np.reshape(masks, (56, 56, -1))
    if masks.shape[-1]<30:
        pad_mask = np.tile(np.zeros((56,56,1)),(1,1,(30-masks.shape[-1])))
        masks = np.concatenate([masks,pad_mask],axis=2)
    return masks, bbox


def create_tfrecord(img_list, full_scale_mask_list):
    # 定义writer,用于向tfrecord写入数据
    write = tf.python_io.TFRecordWriter('mydata.tfrecords')

    for i in tqdm(range(len(img_list))):
        img = cv2.imread(img_list[i], 1)
        img = cv2.resize(img, (500, 500))
        full_scale_mask = cv2.imread(full_scale_mask_list[i], 0)

        masks, bbox = get_single_mask(full_scale_mask)
        # 这里多维的label等转不过去，因此先将label，bbox全部展开，读取后再解析吧
        # bbox = bbox.flatten()
        # labels = labels.flatten()
        # 将图片转为二进制格式
        img = img.tobytes()
        masks = masks.tobytes()
        bbox = np.array(bbox).tobytes()

        # 修改
        # bbox = bbox.astype(np.float32).flatten().tolist()

        # example = tf.train.Example(
        #     features=tf.train.Features(feature={
        #         'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
        #         'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),  # 这里只要保证int64的数据是一个一维list就行了
        #         'masks': tf.train.Feature(bytes_list=tf.train.BytesList(value=[masks])),
        #         'bbox': tf.train.Feature(int64_list=tf.train.Int64List(value=bbox))
        #     })
        # )
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                'masks': tf.train.Feature(bytes_list=tf.train.BytesList(value=[masks])),
                'bboxes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox]))
            })
        )
        write.write(example.SerializeToString())
    write.close()


def read_tfrecord(file_path, flag='train', batch_size=3):
    # 根据文件名生成一个队列
    if flag == 'train':
        filename_queue = tf.train.string_input_producer(file_path)  # 乱序操作并循环读取
    else:
        filename_queue = tf.train.string_input_producer(file_path, num_epochs=1, shuffle=False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,  # 取出包含各个feature
                                       features={
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           # 'labels': tf.FixedLenFeature([], tf.string),
                                           'masks': tf.FixedLenFeature([], tf.string),
                                           'bboxes': tf.FixedLenFeature([], tf.string)
                                       })
    # 将字符串解析成对应的像素数组
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [500, 500, 3])
    masks = tf.decode_raw(features['masks'], tf.uint8)
    masks = tf.reshape(masks,[56,56,-1])
    boxes = tf.decode_raw(features['bboxes'],tf.float32)
    boxes = tf.reshape(boxes,[-1,5])

    # bbox = features['bboxes']


    if flag == 'train':  # 如果训练使用，应当将其归一化并按批次组合
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        masks = tf.cast(masks, tf.float32)
        batch_image, batch_mask, batch_bbox = tf.train.batch([image, masks,boxes],
                                                                          batch_size=batch_size, capacity=20)
        return batch_image, batch_mask, batch_bbox
    return image, masks,boxes


if __name__ == '__main__':
    from tqdm import tqdm
    import tensorflow as tf
    from private_tools.file_opter import get_file_path

    img_dir = r'E:\Pycharm_project\mask_rcnn_TF\voc\val\imgs'
    label_dir = r'E:\Pycharm_project\mask_rcnn_TF\voc\val\labels'
    img_list = sorted(get_file_path(img_dir, ret_full_path=True))
    full_scale_mask_list = sorted(get_file_path(label_dir, ret_full_path=True))
    # 制作tfrecord
    # create_tfrecord(img_list=img_list,full_scale_mask_list=full_scale_mask_list)

    tfrecord_file = r'E:\Pycharm_project\mask_rcnn_TF\mydata.tfrecords'
    batch_image, batch_mask,batch_bbox= read_tfrecord(file_path=[tfrecord_file], flag='test')

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        batch_bbox = batch_bbox.eval()
        batch_image=batch_image.eval()
        batch_mask = batch_mask.eval()

    d = 0
