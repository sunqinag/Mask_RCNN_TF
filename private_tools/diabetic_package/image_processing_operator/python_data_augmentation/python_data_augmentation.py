# ---------------------------------
#   !Copyright(C) 2018,北京博众
#   All right reserved.
#   文件名称：data augmentation
#   摘   要：对图像数据进行增强，包括随机旋转，随机裁剪，随机噪声，平移变换，图像反转,并保存增强后的图像和标签
#   当前版本:2019121111
#   作   者：王茜, 崔宗会
#   完成日期：2019-12-11

import numpy as np
import cv2
import functools
import os
import shutil
import datetime
import filecmp

from . import python_base_data_augmentation
from ...file_operator import bz_path
# from ...data_preprocessing.data_preprocessing import data_preprocessing.load_detection_txt_label, data_preprocessing.save_detection_txt_label
from ...data_preprocessing import data_preprocessing

class DataAugmentation:
    def __init__(self,
                 img_list,
                 label_list,
                 augmentation_ratio,
                 generate_data_folder='./generate_data',
                 channel=3,
                 out_file_extension_list=['jpg', 'npy'],
                 task='classification', augmentation_split_ratio=4):
        """
        对图像和对应标签进行数据增强,若augmentation_ratio小于4,则进行一次数据增强,否则进行
        二次增强,会在原始图像相同的目录下创建两个文件夹augmentation_img和augmentation_label存储增强后的图像和label
        支持分类、分割、目标检测，分类输入的是img_path_list和npy或者txt的label_path_list,
        分割输入的是img_path_list和label_path_list
        目标检测输入的是img_path_list和和npy或者txt的label_path_list，数据为nX5，对于txt存储的标签每行代表一个目标框有５个值，用下划线隔开，依次为
        左上角点的行列右下角点的行列，以及类别
        :param img_list: 输入img_list
        :param label_list: 输入label_list
        :param augmentation_ratio: 增强图像倍数
        :param generate_data_folder:生成文件路径
        :param channel: 图像channel
        :param out_file_extension_list:输出文件格式
        :param task: 增强任务,只能是"classification", "segmentation"和"detection"
        :param augmentation_split_ratio:一次增强和二次增强分割点
        return: 返回增强后的图像和标签的list
        """

        self.img_list = img_list
        self.label_list = label_list
        self.generate_data_folder = generate_data_folder
        self.augmentation_img_list = np.array([])
        self.augmentation_label_list = np.array([])
        self.augmentation_ratio = augmentation_ratio
        self.channel = channel
        self.out_file_extension_list = out_file_extension_list
        self.augmentation_split_ratio = augmentation_split_ratio
        self.task = task.lower()

        self.__check_params()
        self.color_flag = 1
        self.is_enhance_image_only = False
        if self.channel == 1:
            self.color_flag = 0
        self.__init_augmentation_data_fn()

    def augment_data(self):

        if self.is_repeat_data:
            return self._repeat_use_data()
        else:
            self.__create_augmentation_data_dir()
            if self.augmentation_ratio == 1:
                return self._one_ratio_augmentation()
            else:
                return self._multiply_ratio_augmentation()

    def _one_ratio_augmentation(self):
        for i in range(len(self.img_list)):
            img_name, img_extension = bz_path.get_file_name(self.img_list[i], True)

            img_path, label = self.__create_data_fn(self.img_list[i],
                                                    self.label_list[i],
                                                    self.__copy,
                                                    img_name,
                                                    self.out_file_extension_list,
                                                    False)
            self.augmentation_img_list = np.append(
                self.augmentation_img_list, img_path)
            self.augmentation_label_list = np.append(
                self.augmentation_label_list, label)
        np.save(self.data_list_npy_path + '/img.npy', self.augmentation_img_list)
        if self.is_enhance_image_only:
            shutil.rmtree(self.augmentation_label_dir)
            return self.augmentation_img_list
        np.save(self.data_list_npy_path + '/label.npy', self.augmentation_label_list)

        print('数据增强完成！')
        return self.augmentation_img_list, self.augmentation_label_list

    def _multiply_ratio_augmentation(self):
        img_num = len(self.img_list)
        augment_mode = len(self.augment_fn_dict)
        first_augmentation_ratio = np.minimum(self.augmentation_ratio, self.augmentation_split_ratio)
        num = np.int32((first_augmentation_ratio - 1) * img_num / augment_mode)
        remainder = (first_augmentation_ratio - 1) * img_num % augment_mode
        num_list = np.ones(shape=augment_mode, dtype=np.int32) * num
        num_list[0] += remainder
        if self.augmentation_ratio <= self.augmentation_split_ratio:
            # 一次增强
            self.__first_augment_data(num_list)
        else:
            # 一次增强
            self.__first_augment_data(num_list)
            # 二次增强
            second_augmentation_num = np.int64((self.augmentation_ratio - self.augmentation_split_ratio) * img_num)

            self.__second_augment_data(second_augmentation_num)

        if self.is_enhance_image_only:
            self.augmentation_img_list = self._one_ratio_augmentation()
            print('数据增强完成！')
            return self.augmentation_img_list
        # 增加原始图像
        self.augmentation_img_list, self.augmentation_label_list = self._one_ratio_augmentation()

        print('数据增强完成！')
        return self.augmentation_img_list, self.augmentation_label_list

    def __init_augmentation_data_fn(self):
        self.augment_fn_dict = {
            'flip':
                python_base_data_augmentation.random_flip_image_and_label,
            'translate': functools.partial(
                python_base_data_augmentation.random_translation_image_and_label, max_dist=0.15),
            # 'rotate': functools.partial(
            #     python_base_data_augmentation.random_rotate_image_and_label, min_angle=-15, max_angle=15),
            'scale_crop': functools.partial(
                python_base_data_augmentation.random_scale_and_crop_image_and_label,
                min_scale=0.8,
                max_scale=1.2,
                crop_height_scale=0.8,
                crop_width_scale=0.8),
        }
            # 'shear': functools.partial(
            #     python_base_data_augmentation.random_shear_image_and_label, min_scale=0.2, max_scale=-0.2)}
        if self.is_enhance_image_only:
            self.__create_data_fn = \
                self.__create_segmentation_data
        else:
            if self.task == 'classification':
                self.__create_data_fn = \
                    self.__create_classification_data
            elif self.task == 'segmentation':
                self.__create_data_fn = \
                    self.__create_segmentation_data
            elif self.task == 'detection':
                self.__create_data_fn = \
                    self.__create_detection_data

    def __check_params(self):
        if self.label_list is None:
            self.is_enhance_image_only = True
            self.label_list = self.img_list
            self.task = 'segmentation'
            self.out_file_extension_list[1] = self.out_file_extension_list[0]

        if (not isinstance(self.img_list, np.ndarray) or \
                not isinstance(self.label_list, np.ndarray)):
            raise ValueError('img_list和label_list类型必须是np.ndarray!')
        if (len(self.img_list) != len(self.label_list)):
            raise ValueError('原始img和label数量必须相等!')
        if self.augmentation_ratio < 1:
            raise ValueError('增强后图像数量必须大于原始图像数量!')
        if self.channel != 1 and self.channel != 3:
            raise ValueError('图像channel必须是1或者3!')
        if self.channel != 1 and self.channel != 3:
            raise ValueError('图像channel必须是1或者3!')
        if self.task != 'classification' and self.task != 'segmentation' \
                and self.task != 'detection':
            raise ValueError('图像增强任务只能是分类/分割或目标检测!')
        if self.out_file_extension_list[0] not in ['bmp', 'jpg', 'jpeg', 'png']:
            raise ValueError('输出图像目前只支持bmp,jpg,jpeg,png 四种格式！')

        if self.task == 'segmentation' and (self.out_file_extension_list[1] not in ['bmp', 'jpg', 'jpeg', 'png'] or \
                                            self.out_file_extension_list[0] not in ['bmp', 'jpg', 'jpeg', 'png']):
            raise ValueError('对于分割任务，输出图像目前只支持bmp,jpg,jpeg,png 四种格式！')
        if (self.task == 'classification' or self.task == 'objection') and (
            self.out_file_extension_list[0] not in ['bmp', 'jpg', 'jpeg', 'png'] or \
                self.out_file_extension_list[1] not in ['npy', 'txt']):
            raise ValueError("对于分类或分割任务，out_file_extension_list[0]只能为['bmp', 'jpg', 'jpeg', 'png'] 的一种格式，"
                             "out_file_extension_list[1]只能为['npy', 'txt']中的一种格式！")
        self.label_save_function = np.save
        if self.out_file_extension_list[1] == 'txt' and self.task == 'detection':
            self.label_save_function = data_preprocessing.save_detection_txt_label
        elif self.out_file_extension_list[1] == 'txt' and self.task != 'detection':
            self.label_save_function = np.savetxt

        if not os.path.exists(self.generate_data_folder):
            os.makedirs(self.generate_data_folder)

        if os.path.exists(self.generate_data_folder + '/augmentation_information.txt'):
            os.rename(self.generate_data_folder + '/augmentation_information.txt',
                      self.generate_data_folder + '/augmentation_information_old.txt')
            self.__write_parameter_information()
            self.is_repeat_data = filecmp.cmp(self.generate_data_folder + '/augmentation_information.txt',
                                              self.generate_data_folder + '/augmentation_information_old.txt')
            os.remove(self.generate_data_folder + '/augmentation_information_old.txt')
        else:
            self.__write_parameter_information()
            self.is_repeat_data = False

    def __create_augmentation_data_dir(self):
        self.data_list_npy_path = self.generate_data_folder + '/augmentation_data_list_npy/'
        self.augmentation_img_dir = self.generate_data_folder + '/augmentation_img/'
        self.augmentation_label_dir = self.generate_data_folder + '/augmentation_label/'

        if os.path.exists(self.data_list_npy_path):
            shutil.rmtree(self.data_list_npy_path)
        os.mkdir(self.data_list_npy_path)

        if os.path.exists(self.augmentation_img_dir):
            shutil.rmtree(self.augmentation_img_dir)
        os.mkdir(self.augmentation_img_dir)

        if os.path.exists(self.augmentation_label_dir):
            shutil.rmtree(self.augmentation_label_dir)
        os.mkdir(self.augmentation_label_dir)

    def __first_augment_data(self, augmentation_num_list):
        """
        :param augmentation_num_list: 一次增强图像数目list
        :return:
        """
        replace = False
        if len(self.img_list) < np.max(augmentation_num_list):
            replace = True

        for i, (augment_name, augment_fn) in enumerate(
                self.augment_fn_dict.items()):
            num_index = np.random.choice(
                range(len(self.img_list)), augmentation_num_list[i], replace=replace)
            for j in range(augmentation_num_list[i]):
                img_path = self.img_list[num_index][j]
                label = self.label_list[num_index][j]

                origin_img_name, img_extension = bz_path.get_file_name(img_path, True)

                img_name = origin_img_name + '_' + augment_name + '_' + self.__get_timestamp()
                augmentation_img_path, augmentation_label = \
                    self.__create_data_fn(
                        img_path, label, augment_fn, img_name, self.out_file_extension_list, True)
                self.augmentation_img_list = np.append(self.augmentation_img_list, augmentation_img_path)
                self.augmentation_label_list = np.append(self.augmentation_label_list, augmentation_label)

    def __second_augment_data(self, augmentation_num):
        """
        :param augmentation_num: 二次增强图像数目
        :return:返回一次增强后图像和标签的list
        """
        num_index = np.random.choice(range(len(self.augmentation_img_list)),
                                     augmentation_num,
                                     replace=True)
        for i in range(augmentation_num):
            img_path = self.augmentation_img_list[num_index][i]
            label = self.augmentation_label_list[num_index][i]
            random = np.random.randint(0, 3)
            augment_name = list(self.augment_fn_dict.keys())[random]

            origin_img_name, img_extension = bz_path.get_file_name(img_path, True)
            img_name = origin_img_name + '_' + augment_name + '_' + self.__get_timestamp()
            augmentation_img_path, augmentation_label = self.__create_data_fn(
                img_path, label, self.augment_fn_dict[augment_name], img_name, self.out_file_extension_list, True)
            self.augmentation_img_list = np.append(
                self.augmentation_img_list, augmentation_img_path)
            self.augmentation_label_list = np.append(
                self.augmentation_label_list, augmentation_label)

    def __create_segmentation_data(self,
                                   img_path,
                                   label_path,
                                   augment_fn,
                                   img_name,
                                   file_extension_list,
                                   is_adding_noise=False):
        """
        :param img_path: 分割单张图像的路径
        :param label_path: 分割单张label的路径
        :param augment_fn: 增强函数
        :param img_name: 增强后图像的名字
        :param file_extension_list:图像和label的后缀
        :param is_adding_noise: 是否给图像加噪声
        :return:分割增强的图像和标签
        """
        img = cv2.imread(img_path, self.color_flag)
        if is_adding_noise:
            img = self.__random_add_noise_to_img(img)
        label = cv2.imread(label_path, 0)
        augmentation_img, augmentation_label = augment_fn(img, label)

        augmentation_img_path = self.augmentation_img_dir + os.sep + img_name + '.' + file_extension_list[0]
        cv2.imwrite(augmentation_img_path, augmentation_img)
        augmentation_label_path = self.augmentation_label_dir + os.sep + img_name + '.' + file_extension_list[1]

        cv2.imwrite(augmentation_label_path, augmentation_label)
        return augmentation_img_path, augmentation_label_path

    def __create_classification_data(self,
                                     img_path,
                                     label_path,
                                     augment_fn,
                                     img_name,
                                     file_extension_list,
                                     is_adding_noise=False):
        """
        :param img_path: 分类单张图像的路径
        :param label: 分类标签
        :param augment_fn: 增强函数
        :param img_name: 增强后图像的名字
        :param file_extension_list:图像和label的后缀
        :param is_adding_noise: 是否给图像增加噪声
        :return:分类增强的图像和标签
        """
        img = cv2.imread(img_path, self.color_flag)
        if is_adding_noise:
            img = self.__random_add_noise_to_img(img)

        label_ext = bz_path.get_file_name(label_path, True)[1]
        if label_ext == 'txt':
            label = np.loadtxt(label_path)
        else:
            label = np.load(label_path)
        label = int(label)
        augmentation_img, augmentation_label = augment_fn(img, label)
        augmentation_img_path = self.augmentation_img_dir + os.sep + \
            img_name + '.' + file_extension_list[0]

        cv2.imwrite(augmentation_img_path, augmentation_img)
        augmentation_label_path = self.augmentation_label_dir + os.sep + img_name + '.' + file_extension_list[1]
        self.label_save_function(augmentation_label_path, np.array([augmentation_label]))

        return augmentation_img_path, augmentation_label_path

    def __create_detection_data(self,
                                img_path,
                                label_path,
                                augment_fn,
                                img_name,
                                file_extension_list,
                                is_adding_noise=False):
        """
        :param img_path: 分割单张图像的路径
        :param label_path: 分割单张label的路径
        :param augment_fn: 增强函数
        :param img_name: 增强后图像的名字
        :param file_extension_list:图像和label的后缀
        :param is_adding_noise: 是否给图像加噪声
        :return:目标检测增强的图像和标签
        """
        img = cv2.imread(img_path, self.color_flag)
        if is_adding_noise:
            img = self.__random_add_noise_to_img(img)
        label_ext = bz_path.get_file_name(label_path, True)[1]
        if label_ext == 'txt':
            label = data_preprocessing.load_detection_txt_label(label_path)
        else:
            label = np.load(label_path)

        augmentation_img, augmentation_label = augment_fn(img, label)
        augmentation_img_path = self.augmentation_img_dir + os.sep + \
            img_name + '.' + file_extension_list[0]

        cv2.imwrite(augmentation_img_path, augmentation_img)
        augmentation_label_path = self.augmentation_label_dir + os.sep + img_name + '.' + file_extension_list[1]

        self.label_save_function(augmentation_label_path, augmentation_label)

        return augmentation_img_path, augmentation_label_path

    def __copy(self, img, label):
        return img, label

    def __random_add_noise_to_img(self, img):
        """
        :param img: 单张图像
        :return:返回增加噪声后的图像
        """
        noise_mode_list = [['gaussian'],
                           ['salt'],
                           ['s&p'],
                           ['localvar'],
                           ['speckle']]
        mean_var_list = [[0.0001, 0.0001],
                         [0.0001, 0.00005],
                         [0.0001, 0.0001],
                         [0.0001, 0.0001],
                         [0.0001, 0.0001]]
        random_noise = np.random.randint(0, 6)
        if random_noise == 5:
            return img
        else:
            img_noise = python_base_data_augmentation.add_noise(
                img.copy(),
                noise_mode_list[random_noise],
                mean=mean_var_list[random_noise][0],
                var=mean_var_list[random_noise][1])
            return np.uint8(img_noise[0] * 255)

    def _repeat_use_data(self):
        print('增强数据复用！')
        self.data_list_npy_path = self.generate_data_folder + '/augmentation_data_list_npy/'
        self.augmentation_img_list = np.load(
            self.data_list_npy_path + 'img.npy')
        if self.is_enhance_image_only:
            return self.augmentation_img_list

        self.augmentation_label_list = np.load(
            self.data_list_npy_path + 'label.npy')
        return self.augmentation_img_list, self.augmentation_label_list

    def __get_timestamp(self):
        ts = str(int(1000 * datetime.datetime.now().timestamp()))
        return ts[-6:]

    def __write_parameter_information(self):
        augmentation_information_file = open(self.generate_data_folder + '/augmentation_information.txt', 'w+')
        augmentation_information_file.write('augmentation_ratio=' + str(self.augmentation_ratio) + '\n')
        augmentation_information_file.write('generate_data_folder=' + self.generate_data_folder + '\n')
        augmentation_information_file.write('channel=' + str(self.channel) + '\n')
        augmentation_information_file.write('out_file_extension_list=' + str(self.out_file_extension_list) + '\n')
        augmentation_information_file.write('task=' + self.task + '\n')
        augmentation_information_file.write('img_list=' + str(self.img_list) + '\n')
        augmentation_information_file.write('label_list=' + str(self.label_list) + '\n')
        augmentation_information_file.close()


# if __name__ == '__main__':
#     from diabetic_package.file_operator import bz_path
#     import matplotlib.pyplot as plt
#
#     # 分割路径
#     img_dir = '/home/bz/PycharmProjects/phone/indus_data_base/1/imgs'
#     label_dir = '/home/bz/PycharmProjects/phone/indus_data_base/1/labels'
#     img_list = np.sort(bz_path.get_file_path(img_dir, ret_full_path=True))
#     label_list = np.sort(bz_path.get_file_path(label_dir, ret_full_path=True))
#     generate_data_folder='/home/wx/git_code/20191029/out/'
#     # # 分类路径
#     # img_dir = '/home/wx/data/PycharmProjects/segmentation_detection/Untitled Folder/pic'
#     # img_list = np.sort(bz_path.get_file_path(img_dir, ret_full_path=True))
#     # label_list = np.ones(shape=len(img_list), dtype=int)
#
#     augmentation_data = DataAugmentation(img_list=img_list,
#                                          label_list=label_list,
#                                          channel=3,
#                                          augmentation_ratio=3,
#                                          generate_data_folder=generate_data_folder,
#                                          task = 'segmentation')
#     augmentation_img_list, augmentation_label_list = augmentation_data.augment_data()
