# ----------------------------
# !  Copyright(C) 2019
#   All right reserved.
#   文件名称：python_split_train_eval_test_data.py
#   摘   要：数据前处理模块
#   当前版本:2019120317
#   作   者：刘恩甫,崔宗会
#   完成日期：2019-12-03
# -----------------------------

import os
import shutil
import numpy as np
import functools

from diabetic_package.file_operator.bz_path import get_file_name, get_all_subfolder_img_label_path_list
from diabetic_package.file_operator.balance_data import BalanceData


def get_preprocessing_img_and_label_path_list(
        base_folder,
        generate_data_folder='./generate_data/',
        max_augment_num=None,
        is_balance=True,
        mode='train', task='classification', out_file_extension_list=['jpg', 'txt']):
    """
    获取前处理后的img和label 列表，可以进行balance也可以不进行balance
    :param base_folder:数据路径,其下面包含如1,2,3,4等文件夹，1，2,3,4下包含img，label文件夹
    :param generate_data_folder:生成数据路径
    :param max_augment_num:均衡化后每个类别数目，如果为None则模型按类别最多的数目
    :param is_balance:是否进行均衡
    :param mode:模式，"只能输入train，val，test，TRAIN，VAL，TEST，Train，Val，Test"
                         "中的一个！"
    :return:img，label 的list
    """
    if not os.path.exists(base_folder):
        raise ValueError("路径" + base_folder + "不存在！")
    mode_lower = mode.lower()
    if mode_lower not in ['train', 'val', 'test']:
        raise ValueError(
            "mode 只能输入train，val，test，TRAIN，VAL，TEST，"
            "Train，Val，Test中的一个！")
    # generate_data_folder_new = generate_data_folder + os.sep + mode_lower + os.sep
    # # if not os.path.exists(generate_data_folder_new):
    # #     os.makedirs(generate_data_folder_new)

    if is_balance:
        balance_obj = BalanceData(base_folder=base_folder,
                                  generate_data_folder=generate_data_folder,
                                  max_augment_num=max_augment_num, task=task,
                                  out_file_extension_list=out_file_extension_list)
        img_list, label_list = balance_obj.create_data()

        # img_list, label_list = balance_data(
        #     base_folder=base_folder,
        #     generate_data_folder=generate_data_folder_new,
        #     max_augment_num=max_augment_num,task=task)

        return img_list, label_list
    else:
        img_list, label_list = get_all_subfolder_img_label_path_list(
            base_folder, ret_full_path=True)
        if mode == 'test' or mode == 'validate':
            return img_list, label_list

        shuffle_indices = np.arange(len(img_list))
        np.random.shuffle(shuffle_indices)
        shuffle_img_list = img_list[shuffle_indices]
        shuffle_label_list = img_list[shuffle_indices]
        return shuffle_img_list, shuffle_label_list


def split_train_eval_test_data(img_list, label_list=None, train_folder='split_train_eval_test' + os.sep + 'train',
                               eval_folder='split_train_eval_test' + os.sep + 'eval',
                               test_folder='split_train_eval_test' + os.sep + 'test',
                               ratio=(0.6, 0.1, 0.3), task='Segmentation'):
    '''
    对于image,label数据，实现数据集的按比例自动划分。
    :param img_list: 待划分图像的全路径列表,list或ndarray类型
    :param label_list:待划分标签的全路径列表,list或ndarray或None，如果为None,则只进行img的划分，不进行label的划分
    :param ratio:train,eval,test的分割比例，其加和值严格为1，且每个元素取值为[0,1]
    :param task:任务类型，包括分类、分割、目标检测，默认为分割
    :param train_folder: 划分后的train数据保存文件夹，会在其内部自动创建img、label文件夹，如果没有指定则使用默认路径
    :param eval_folder: 划分后的eval数据保存文件夹，会在其内部自动创建img、label文件夹，如果没有指定则使用默认路径
    :param test_folder: 划分后的test数据保存文件夹，会在其内部自动创建img、label文件夹，如果没有指定则使用默认路径
    输出目录结构为：

    train_folder
        |______img
        |______label


    eval_folder
        |______img
        |______label


    test_folder
        |______img
        |______label
    '''

    # 参数检查部分
    img_list, label_list, ratio, task = __check_param(img_list, label_list, ratio, task)

    # train,eval,test文件夹的创建,out_path是输出路径的集合,[img_path,label_path]*(train,eval,test)格式
    out_path = __mkdirs(train_folder, eval_folder, test_folder)

    # 计算shuffle的索引
    total_num = len(img_list)
    shuffle_index = np.arange(total_num)
    np.random.shuffle(shuffle_index)

    # 计算img_list的、label_list的分段
    train_eval_test_num = [0, int(ratio[0] * total_num), int((ratio[0] + ratio[1]) * total_num), total_num]
    split_img_list = [img_list[shuffle_index[train_eval_test_num[i]:train_eval_test_num[i + 1]]] \
                      for i in np.arange(len(train_eval_test_num) - 1)]

    split_label_list = []
    if label_list is not None:
        split_label_list = [label_list[shuffle_index[train_eval_test_num[i]:train_eval_test_num[i + 1]]] \
                            for i in np.arange(len(train_eval_test_num) - 1)]

    # 进行划分
    if label_list is None:  # 无标签只划分图片的情况
        for i in np.arange(len(split_img_list)):
            __img_copy(split_img_list[i], out_path[i][0])
    elif task in ('classification', 'detection'):
        for i in np.arange(len(split_img_list)):
            __img_copy(split_img_list[i], out_path[i][0])
            __npy_save(split_img_list[i], split_label_list[i], out_path[i][1])
    else:  # segmentation任务的划分
        for i in np.arange(len(split_img_list)):
            __img_copy(split_img_list[i], out_path[i][0])
            __img_copy(split_label_list[i], out_path[i][1])


def load_detection_txt_label(detection_txt_label_path):
    if not os.path.exists(detection_txt_label_path):
        raise ValueError("路径(" + detection_txt_label_path + ")对应txt文件不存在!")
    _, ext = get_file_name(detection_txt_label_path, True)
    if ext != 'txt':
        raise ValueError("路径(" + detection_txt_label_path + ")不是txt文件!")

    label = []
    file = open(detection_txt_label_path, 'r+')
    for line in file.readlines():
        line = line[:-1].split('_')
        line = list(map(lambda x: float(x), line))
        label.append(line)
    label = np.array(label)
    return label


def save_detection_txt_label(detection_txt_label_path, label):
    _, ext = get_file_name(detection_txt_label_path, True)
    if ext != 'txt':
        raise ValueError("路径(" + detection_txt_label_path + ")不是txt文件!")

    if os.path.exists(detection_txt_label_path):
        os.remove(detection_txt_label_path)

    with open(detection_txt_label_path, 'w+') as f:
        for one_label in label:
            one_label_str = functools.reduce(lambda x, y: x + y, [str(i) + '_' for i in one_label])
            f.write(one_label_str[:-1] + '\n')


def __check_param(img_list, label_list, ratio, task):
    # img_list,label_list的参数检查
    if not isinstance(img_list, list) and not isinstance(img_list, np.ndarray):
        raise ValueError("img_list的类型为list或者ndarray类型!")

    if len(img_list) == 0:
        raise ValueError("img_list是空列表，请检查!")

    if not os.path.exists(img_list[0]):
        raise ValueError("img_list中的文件不存在，可能是非全路径，请检查！")

    if isinstance(img_list, list):
        img_list = np.array(img_list)

    # 对label_list的参数检查
    if label_list is not None:
        # list类型转换ndarray
        if isinstance(label_list, list):
            label_list = np.array(label_list)

        if not isinstance(label_list, list) and not isinstance(label_list, np.ndarray):
            raise ValueError("label_list的类型为list或者ndarray类型！")

        # img_list,label_list的长度检查
        if len(img_list) != len(label_list):
            raise ValueError("img_list,label_list的长度需要保持一致！")

    # ratio的参数检查
    ratio = np.array(ratio)
    if ratio.sum() != 1 or len(ratio) != 3 or np.any(ratio < 0) or np.any(ratio > 1):
        raise ValueError(" 'ratio'元素取值为[0,1],其加和需为1，请输入正确比例或取值！")

    # 任务类型的参数检查
    task = task.lower()
    if task not in ('classification', 'segmentation', 'detection'):
        raise ValueError("task　必须为'classification','Classification','CLASSIFICATION','segmentation',"
                         "'SEGMENTATION','Segmentation','detection','detection'")

    # segmentation特有的img_list,label_list的一致性检查(长度一样，但是名称不一致的情况)
    if task == 'segmentation':
        check_img_list = list(map(lambda x: get_file_name(x), img_list))
        check_label_list = list(map(lambda x: get_file_name(x), label_list))
        diff = list(set(check_img_list) ^ set(check_label_list))
        if len(diff) != 0:
            raise ValueError('img_list、label_list必须保持对应关系！')

    return img_list, label_list, ratio, task


def __mkdirs(train_folder, eval_folder, test_folder):
    # 创建3对[img,label]，到对应的输出路径
    type_key = ['img', 'label']
    out_path = []

    for folder in [train_folder, eval_folder, test_folder]:
        temp_list = []
        for tpk in type_key:
            if os.path.exists(folder + os.sep + tpk):
                shutil.rmtree(folder + os.sep + tpk)
            temp_list.append(folder + os.sep + tpk + os.sep)
            os.makedirs(folder + os.sep + tpk + os.sep)
        out_path.append(temp_list)
    return out_path


def __img_copy(src_path_list, dst_path):
    for img in src_path_list:
        shutil.copy(img, dst_path + '.'.join(get_file_name(img, return_ext=True)))


def __npy_save(src_img_path_list, src_label_path_list, dst_path):
    for img, label in zip(src_img_path_list, src_label_path_list):
        # np.save(dst_path + ''.join(get_file_name(img, return_ext=False)) + '.npy', label)
        shutil.copy(label, dst_path + ''.join(get_file_name(img, return_ext=False)) + '.txt')
