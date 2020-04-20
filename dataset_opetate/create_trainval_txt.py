'''
按照voc数据集的训练要求，要将图像位置以及对应的xml文件写入一个txt文件，统训练时读取，保存为trainval.txt
'''
# ! /usr/bin/python
# -*- coding:UTF-8 -*-

import os, sys
import glob

trainval_image_dir = '/media/bzjgsq/6000cd6e-6c7d-4f5d-b5e3-1d2073ae9f32/caffe-jacinto/data/dataset/VOCdevkit/barcode_img/voc/train/images'
trainval_anno_dir = '/media/bzjgsq/6000cd6e-6c7d-4f5d-b5e3-1d2073ae9f32/caffe-jacinto/data/dataset/VOCdevkit/barcode_img/voc/train/Annotations'
test_image_dir = '/media/bzjgsq/6000cd6e-6c7d-4f5d-b5e3-1d2073ae9f32/caffe-jacinto/data/dataset/VOCdevkit/barcode_img/voc/test/images'
test_anno_dir = '/media/bzjgsq/6000cd6e-6c7d-4f5d-b5e3-1d2073ae9f32/caffe-jacinto/data/dataset/VOCdevkit/barcode_img/voc/test/Annotations'

image_format = '.bmp'

trainval_img_lists = glob.glob(trainval_image_dir + '/*' + str(image_format))
trainval_img_names = []
for item in trainval_img_lists:
    temp1, temp2 = os.path.splitext(os.path.basename(item))
    trainval_img_names.append(temp1)

test_img_lists = glob.glob(test_image_dir + '/*' + str(image_format))
test_img_names = []
for item in test_img_lists:
    temp1, temp2 = os.path.splitext(os.path.basename(item))
    test_img_names.append(temp1)

trainval_fd = open(
    "/media/bzjgsq/6000cd6e-6c7d-4f5d-b5e3-1d2073ae9f32/caffe-jacinto/data/dataset/VOCdevkit/barcode_img/voc/trainval.txt",
    'w')
test_fd = open(
    "/media/bzjgsq/6000cd6e-6c7d-4f5d-b5e3-1d2073ae9f32/caffe-jacinto/data/dataset/VOCdevkit/barcode_img/voc/test.txt",
    'w')

for item in trainval_img_names:
    print(trainval_image_dir + '/' + str(item) + '.png' + ' ' + trainval_anno_dir + '/' + str(item) + '.xml\n')
    trainval_fd.write(
        trainval_image_dir + '/' + str(item) + str(image_format) + ' ' + trainval_anno_dir + '/' + str(item) + '.xml\n')

for item in test_img_names:
    test_fd.write(
        test_image_dir + '/' + str(item) + str(image_format) + ' ' + test_anno_dir + '/' + str(item) + '.xml\n')
