# ------------------------------------------------
#   !Copyright(C) 2018, 北京博众
#   All right reserved.
#   文件名称：
#   摘   要: 提取眼科图像红色病变区域候选区
#   当前版本 : 0.0
#   作   者 ：于川汇 陈瑞侠
#   完成日期 : 2018-1-26
# ------------------------------------------------

import cv2
import numpy as np
from ..image_processing_operator import struct_element
from ..image_processing_operator import region_operator, bz_morphology


def get_mask(img):
    '''
        作用：
            获取二维图像的mask区域
        参数：
            img： 输入图像
    '''
    r, g, b = cv2.split(img)
    mean = np.mean(r)
    if mean > 45:
        threhold = np.mean(r) / 3 - 10
    else:
        threhold = 5
    _, mask = cv2.threshold(r, threhold, 1, cv2.THRESH_BINARY)
    mask = bz_morphology.fill_hole(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    regions = region_operator.region_props(mask)
    if len(regions) != 0:
        region_selector = region_operator.RegionSelector(regions)
        region_selected_label = region_selector.select_region_max('area')
        mask = region_selector.selected_region_binary(region_selected_label, mask.shape, 0, 1)
        return np.uint8(mask)
    else:
        return np.uint8(np.ones(r.shape))


def get_Iw(img, mask):
    '''
       作用：
           获取r-polynomial转换后的图像
       参数：
           img： 输入图像
       方法：
          1、拆分3通道，获得绿色通道图像G
          2、针对图像边缘进行扩充，防止处理过程中边缘病灶漏检
          3、对图像进行mask区域过滤
          4、先求G图像的W邻域内的均方滤波图像uw
          5、进行r-polynomial转换
             （1）求绿色通道图像的最大值G_max和最小值G_min
             （2）对于像素值大于G的图像：
                  Iw (i, j ) = (1/2*(u_max-u_min)/(uw(i, j)-G_min)**r)*(G - G_min) ** r + u_min
             （3）对于像素值小于G的图像：
                  Iw (i, j ) = (1/2*(u_max-u_min)/(uw(i, j)-G_max)**r)*(G - G_max) ** r + u_max
             （4）对图像进行高斯滤波去噪
              特殊情况：
                    当出血面积过大时或者图像区域较暗时，u_max-u_min的值可能为0，针对
                u_max-u_min差值为0的情况做以下特殊处理：
                （1）先获得矩阵中不为0的部分u_min_nozero
                （2）获得矩阵中为0的部分u_mins_zero
                （3）将不为0的部分和为0的部分进行叠加得到u_mins_add，将叠加的结果代入3（2）的公式中求Iw(i, j)
                （4）求u_min_nozero与Iw(i, j)的交集，最终得到的Iw(i,j)

    '''
    DIRIVE_kernel_size = 25
    DIRIVE_h_resolution = 536
    W = int((DIRIVE_kernel_size * len(img[0])) / DIRIVE_h_resolution)
    u_max = 1
    u_min = 0
    r = 2
    G = img[:, :, 1]
    G = np.multiply(G, mask)
    G = G / np.max(G)
    G_min, G_max, *_ = cv2.minMaxLoc(G, np.uint8(mask))
    uw = cv2.blur(G, (W, W))
    u_mins = (uw - G_min)
    u_min_nozero = (np.greater(u_mins, 0)).astype('int')
    u_mins_zero = 1 - u_min_nozero
    u_mins_add = u_mins + u_mins_zero
    rt_low = (1 / 2 * (u_max - u_min) / (u_mins_add ** r + 1e-7)) \
        * (G - G_min) ** r + u_min
    rt_low_min = np.multiply(u_min_nozero, rt_low)
    rt_high = (-1 / 2 * (u_max - u_min) / ((uw - G_max) ** r + 1e-7)) \
        * (G - G_max) ** r + u_max
    uw_more_than_g = np.multiply(np.float32(uw >= G), rt_low_min)
    uw_less_than_g = 1 - np.greater_equal(uw, G)
    uw_less_than_g = np.multiply(uw_less_than_g, rt_high)
    Iw = uw_more_than_g + uw_less_than_g
    Iw_gauss = cv2.GaussianBlur(Iw, (5, 5), 1, 1)
    Iw_mask = np.multiply(Iw_gauss, mask)
    return Iw_mask


def _threshold_cand_l(I, K):
    '''
        作用：
            获取经过混合阈值处理后的二值图像
        参数
            I： 输入图像
            K： 候选区域的目标值
    '''

    max_num_conn_comp = 0
    I_min, I_max, *_ = cv2.minMaxLoc(I)
    t_l = 0.005
    t_u = 0.2
    t = 0
    increment = 0.003
    px = 5
    search_steps = np.floor((I_max - I_min) / increment)

    while (max_num_conn_comp <= K) and (t <= search_steps):
        t_s = I_max - increment * t
        binary = np.uint8(np.greater_equal(I, t_s))
        binary, contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_num_conn_comp = len(contours)
        t += 1
    if (t == search_steps and max_num_conn_comp < K):
        # 当t==search_steps并且max_num_conn_comp<K时说明遍历所有的t_s,
        # max_num_conn_comp都小于K,t_k=t_l
        t_k = t_l
    elif (t < search_steps and max_num_conn_comp > K):
        t_k = search_steps * increment - increment * (t - 2)

    else:
        # 当t==1并且max_num_conn_compK时说明遍历所有的t_s,
        # max_num_conn_comp都大于K,t_k=t_u
        t_k = t_u

    cand = np.uint8(np.greater_equal(I, t_k))
    cand_remove_small_size = _remove_small_area(cand, px)
    return np.uint8(cand_remove_small_size)


def _remove_small_area(img, min_area):
    '''
        作用：
            删除二直图像中面积小于min_area的区域
        参数:
            img     ： 输入的二直图像
            min_area： 最小面积
    '''
    regions = region_operator.region_props(img)
    region_selector = region_operator.RegionSelector(regions)
    region_selected_label = region_selector.select_region(
        ['area'], [min_area], [np.inf], 'and')
    img_removed_small_area = region_selector.selected_region_binary(
        region_selected_label, img.shape, 0, 1)
    return img_removed_small_area


def _get_I_cand_l(I, l):
    '''
        作用：
            获取l尺度下的candidate图像
        参数:
            I： 输入图像
            l： 形态学closing的线型kernel大小
    '''
    closed_images = []
    max_search_angle = 180
    delta_angle = 15
    for deg in range(0, max_search_angle, delta_angle):
        kernel = struct_element.get_line_kernel(l, deg)
        closed_images.append(cv2.morphologyEx(I, cv2.MORPH_CLOSE, kernel))
    return np.min(closed_images, 0) - I


def get_candidates(img, mask):
    '''
        作用：
           获取图像病变候选区域
        参数:
            img： 输入图像
    '''

    Iw_new = get_Iw(img, mask)
    K = 120
    bw_l = []
    delt = img.shape[1] * 5 / 1924
    scale = 45
    for L in np.arange(20, scale, delt):
        cand_l = _get_I_cand_l(Iw_new, L)
        I_closed = _threshold_cand_l(cand_l, K)
        bw_l.append(I_closed)
    small_candidates = np.amax(bw_l, axis=0)
    return small_candidates


def enhance_img(I, mask):
    '''
        作用：
            对比度均衡化处理
        参数：
            I   ：输入图像
            mask：掩码区域

    '''

    w = int(3 * len(I[0]) / 30)
    sigma = len(I) / 30
    if w % 2 == 0:
        w = w + 1
    b, g, r = cv2.split(I)
    b_pad = _img_pad(w, sigma, mask, b)
    g_pad = _img_pad(w, sigma, mask, g)
    r_pad = _img_pad(w, sigma, mask, r)
    I_out = cv2.merge([b_pad, g_pad, r_pad])

    return I_out


def _img_pad(w, sigma, mask, img):
    img_background = np.uint8(np.multiply(img, mask) == 0)\
        * (np.max(cv2.mean(img, mask)) - 70)
    img_pad = img_background + img
    I_gauss = cv2.GaussianBlur(img_pad, (w, w), sigma)
    I_extended = 4 * img_pad - 4 * I_gauss + 128
    mean_min, mean_max = _get_mean_min_and_max(I_extended, percent=0.005)
    I_current = (I_extended - mean_min) / (mean_max - mean_min) * 255
    I_current = np.multiply(I_current, mask)
    I_current = np.multiply(I_current > 255, 255) + \
        np.multiply(I_current < 255, I_current)
    I_current = np.multiply(I_current < 0, 0) + \
        np.multiply(I_current > 0, I_current)
    return I_current


def _get_mean_min_and_max(I, percent):
    if percent == 0:
        mean_min = np.min(I)
        mean_max = np.max(I)
    else:
        data_num = np.prod(I.shape)

        I_1d = np.reshape(I, data_num)
        find_num = np.int(np.floor((data_num * percent)))
        I_1d = np.sort(I_1d)
        mean_max = np.mean(I_1d[-find_num:])
        mean_min = np.mean(I_1d[: find_num])
    return mean_min, mean_max


if __name__ == '__main__':
    from diabetic_package.file_operator import bz_path
    import time
    import os
    data_path = './dr1_test/'
    result_path = '/home/crx/crx/git_code/test_scale_and_threshold/mask_img/'
    data_list = bz_path.get_file_path(data_path)
    num_data = len(data_list)
    for i in range(num_data):
        start_time = time.time()
        img = cv2.imread(data_path + data_list[i])
        print(img.shape)
        # candidates, mask, Iw = get_candidates(img)
        # print('enhance_time', time.time() - time1)
        mask = get_mask(img)
        # img_mask = img * mask
        # mask1 = get_img_mask(img)
        # time1 = time.time()
        # en_img = enhance_img(img, mask)
        #
        img_id = os.path.splitext(data_list[i])[0]
        img_ext = os.path.splitext(data_list[i])[1]
        #
        # cv2.imwrite(result_path  +  img_id + img_ext , mask)
        cv2.imwrite(result_path + img_id + '__' + img_ext, mask)
        # # cv2.imwrite(result_path + img_id +'__' + img_ext, en_img)
        # print(time.time() - start_time)
