# -----------------------------------------------------------------------
#   !Copyright(C) 2018, 北京博众
#   All right reserved.
#   文件名称：
#   摘   要: 根据提取的候选区域regions，从原始图像中提取自定义大小的图像小块
#   当前版本 : 0.0
#   作   者 ：于川汇 陈瑞侠
#   完成日期 : 2018-2-2
# -----------------------------------------------------------------------

import numpy as np
import cv2

def _default_win_size_func(major_axis_length, window_size):
    '''
        作用：
            根据输入的region区域的面积和自定义窗口尺寸window_size的大小
            获得region区域的裁切窗口大小
        参数:
            region     ： 输入的候选区域的region
            window_size： 自定义窗口的大小
    '''
    if (major_axis_length * 2 < window_size):
        current_window_size = window_size
    else:
        current_window_size = np.ceil(major_axis_length) * 2
    return current_window_size

def extract_patch(img,
                  window_size=32,
                  window_size_func=_default_win_size_func,
		          **kwargs):
    '''
        参数：
            img             : 增强后的图像, 图像数据类型: uint8, 图像数据范围[0,255]
            regions         : 候选区域列表
            window_size     : 窗口大小
            window_size_func: 用户定制patch窗口大小的回调函数,
                              它的第一个参数是region，第二个参数
                              是期望获得的patch的窗口尺寸
        返回值：
            ma_windows : 根据自适应窗口大小裁切的图像小块

    '''
    centroids = kwargs['centroid']
    major_axis_lengths = kwargs['major_axis_length']
    patches = np.zeros(
        shape=[
            len(centroids),
            window_size,
            window_size,
            img.shape[2]],
        dtype=img.dtype)

    for k in range(len(centroids)):
        current_window_size = window_size_func(major_axis_lengths[k], window_size)
        center = np.round(centroids[k])
        half_current_window_size = np.ceil(current_window_size / 2).astype(int)
        bbox = [
            center[0] - half_current_window_size,
            center[1] - half_current_window_size,
            center[0] + half_current_window_size,
            center[1] + half_current_window_size
        ]
        current_window_size = half_current_window_size * 2
        bbox = np.ceil(bbox).astype(np.int)
        current_window = np.zeros([current_window_size,
                                   current_window_size,
                                   img.shape[2]], dtype=img.dtype)
        c_r1 = 0
        c_r2 = current_window_size
        c_c1 = 0
        c_c2 = current_window_size
        b_r1 = bbox[0]
        b_r2 = bbox[2]
        b_c1 = bbox[1]
        b_c2 = bbox[3]
        if bbox[0] < 0:
            c_r1 = -bbox[0]
            b_r1 = 0
        if bbox[1] < 0:
            c_c1 = -bbox[1]
            b_c1 = 0
        if bbox[2] > img.shape[0]:
            c_r2 = current_window_size - (bbox[2] - img.shape[0])
            b_r2 = img.shape[0]
        if bbox[3] > img.shape[1]:
            c_c2 = current_window_size - (bbox[3] - img.shape[1])
            b_c2 = img.shape[1]
        current_window[c_r1:c_r2, c_c1:c_c2] = img[b_r1:b_r2, b_c1:b_c2]

        if current_window_size > window_size:
            patches[k] =cv2.resize(current_window,(window_size, window_size))
        else:
            patches[k] = current_window

    return patches



def _rect_teansform(window_size,window_center,img_shape):
    '''
    函数将图片中的位置，转化为patch的位置
    输入值：
        window_size:    窗口的尺寸
        window_center:  窗口的中心(int)
    返回值:
        (img_rect, window_rect)
        img_rect        : 图片中窗口的坐标信息
        window_rect     : patch的坐标信息        
    '''
    half_window_size = int(window_size / 2)
    bbox = [
        np.int(window_center[0]) - half_window_size,
        np.int(window_center[1]) - half_window_size,
        np.int(window_center[0]) - half_window_size+window_size,
        np.int(window_center[1]) - half_window_size+window_size]
    img_rect=Rect()
    window_rect=Rect()
    window_rect.c_start=0
    window_rect.c_end=window_size
    window_rect.r_start=0
    window_rect.r_end=window_size
    img_rect.c_start=bbox[1]
    img_rect.c_end=bbox[3]
    img_rect.r_start=bbox[0]
    img_rect.r_end=bbox[2]
    if img_rect.r_start < 0:
        window_rect.r_start = -img_rect.r_start
        img_rect.r_start = 0
    if img_rect.c_start < 0:
        window_rect.c_start = -img_rect.c_start
        img_rect.c_start = 0
    if img_rect.r_end > img_shape[0]:
        window_rect.r_end = window_size - (img_rect.r_end - img_shape[0])
        img_rect.r_end = img_shape[0]
    if img_rect.c_end > img_shape[1]:
        window_rect.c_end = window_size - (img_rect.c_end - img_shape[1])
        img_rect.c_end = img_shape[1]
    return (window_rect,img_rect)

def _get_image_patch(img, window_center, window_size, patch_size):
    '''
    从图片中扣取指定位置、大小的图像块，把图像块转化为指定大小的图像，并返回
    输入值：
        img             :图像
        window_center   :图像块的中心
        window_size     :图像块的尺寸
        patch_size      :返回图像的尺寸
    返回值：
        img_result      :转化后的图像
    '''
    img_result=np.zeros(
        shape=[window_size,window_size,img.shape[2]],
        dtype=img.dtype)
    w_r,i_r=_rect_teansform(window_center=window_center,window_size=window_size,img_shape=img.shape)
    img_result[w_r.r_start:w_r.r_end,w_r.c_start:w_r.c_end,:]=img[i_r.r_start:i_r.r_end,i_r.c_start:i_r.c_end,:]
    return cv2.resize(src=img_result,dsize=(patch_size,patch_size))

def get_pyramid_data(regions,img,patch_size=32,pyramid=(2,4,8)):
    '''
    从图像中获取region的金字塔数据（可用于神经网络的训练和测试）
    分别提取2倍、4倍、8倍max_axis_length大小的regions处图像
    输入参数：
        regions:区域参数
        img:图像
        patch_size:patch的大小
    输出参数：
        result:数据 len(regions)*32*32*img.shape[2]*3
    '''
    result=np.zeros(
        shape=[len(regions),patch_size,patch_size,img.shape[2]*len(pyramid)],
        dtype=img.dtype)
    for i in range(len(regions)):
        r=regions[i]
        for j in range(len(pyramid)):#j 遍历
            p=pyramid[j]
            tmp_patch=_get_image_patch(img=img,window_center=r.centroid,window_size=np.int(r.major_axis_length*p),patch_size=patch_size)
            result[i,:,:,j*img.shape[2]:(j+1)*img.shape[2]]=tmp_patch
        
    return result


class Rect(object):
    def __init__(self):
        self.c_start=0
        self.c_end=0
        self.r_start=0
        self.r_end=0



# if __name__=='__main__':
#     file_path='./000010944_20140407112750_900421_69877.npz'
#     loaded=np.load(file_path)
#     regions=loaded['region']
#     img=loaded['img_eq']
#     data=get_pyramid_data(regions,img)




