'''
test_name_size.txt 的文件，里面记录训练图像、测试图像的图像名称、height、width
'''
# ! /usr/bin/python

import os, sys
import glob
from PIL import Image

img_dir = "/media/bzjgsq/6000cd6e-6c7d-4f5d-b5e3-1d2073ae9f32/caffe-jacinto/data/dataset/VOCdevkit/barcode_img/voc/test/images/"

img_lists = glob.glob(img_dir + '/*.bmp')

test_name_size = open(
    '/media/bzjgsq/6000cd6e-6c7d-4f5d-b5e3-1d2073ae9f32/caffe-jacinto/data/dataset/VOCdevkit/barcode_img/voc/test/test_name_512_size.txt',
    'w')

for item in img_lists:
    img = Image.open(item)
    width, height = img.size
    temp1, temp2 = os.path.splitext(os.path.basename(item))
    test_name_size.write(temp1 + ' ' + str(height) + ' ' + str(width) + '\n')
