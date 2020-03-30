import cv2
import numpy as np
import matplotlib.pyplot as plt
from private_tools.imgFileOpterator import Img_processing

def get_rpb_label(src_img,src_label,down_sample_ratio):
    '''
    通过使用原始图像和box坐标计算出输入rpn层的feature map的尺寸和对应的anchor的正确位置
    :param src_img: 给入 已经解析好的img
    :param src_label: 给入原始的label路径，会在内部解析
    :param down_sample_ratio: 下采样比率
    :return:
    '''
    src_height,src_width=src_img.shape[0],src_img.shape[1]
    img = cv2.resize(src_img,(int(src_width/down_sample_ratio),int(src_height/down_sample_ratio)))
    rpn_height,rpn_width=img.shape[0],img.shape[1]

    rpn_match = np.zeros((rpn_height,rpn_width))
    box,_ = Img_processing().parseBoxAndLabel(src_label)

    #将box找出acnhor点，再缩小到rpn所接收到的下采样后的尺寸
    for cor in box:
        width,height = cor[2]-cor[0],cor[3]-cor[1]
        center = [cor[0]+width/2,cor[1]+height/2]
        rpn_match[int(center[1]/down_sample_ratio),int(center[0]/down_sample_ratio)]=1
        img[int(center[1]/down_sample_ratio),int(center[0]/down_sample_ratio),:]=255
    plt.figure(figsize=(6,6))
    plt.imshow(rpn_match)
    plt.show()
    return rpn_match

def get_single_mask(src_img,src_label):
    bbox,label=Img_processing().parseBoxAndLabel(src_label)
    height,width=src_img.shape[0],src_img.shape[1]
    for box,label in zip(bbox,label):
        single_mask = np.zeros((height,width))
        single_mask[box[0]:box[2],box[1]:box[3]]=label
        plt.figure(figsize=(6,6))
        plt.imshow(single_mask)
        plt.show()


if __name__ == '__main__':
    import tensorflow as tf
    image_path = '0822/img1.bmp'
    label_path = '0822/img1.txt'
    down_sample = 6  # 主干网络会下采样6倍
    src_img = cv2.imread(image_path,1)
    get_single_mask(src_img,label_path)

