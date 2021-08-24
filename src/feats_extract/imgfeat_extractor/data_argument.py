import random
import cv2
import numpy as np


def horizon_flip(img):
    '''
    图像水平翻转
    :param img:
    :return:水平翻转后的图像
    '''
    return img[:, ::-1]


def vertical_flip(img):
    '''
    图像垂直翻转
    :param img:
    :return:
    '''
    return img[::-1]


def rotate(img, limit_up=10, limit_down=-10):
    '''
    在一定角度范围内，图像随机旋转
    :param img:
    :param limit_up:旋转角度上限
    :param limit_down: 旋转角度下限
    :return: 旋转后的图像
    '''
    # 旋转矩阵
    rows, cols = img.shape[:2]
    center_coordinate = (int(cols / 2), int(rows / 2))
    angle = random.uniform(limit_down, limit_up)
    M = cv2.getRotationMatrix2D(center_coordinate, angle, 1)

    # 仿射变换
    out_size = (cols, rows)
    rotate_img = cv2.warpAffine(img, M, out_size, borderMode=cv2.BORDER_REPLICATE)

    return rotate_img


def shift(img, distance_down, distance_up):
    '''
    利用仿射变换实现图像平移，平移距离∈[down, up]
    :param img: 原图
    :param distance_down:移动距离下限
    :param distance_up: 移动距离上限
    :return: 平移后的图像
    '''
    rows, cols = img.shape[:2]
    y_shift = random.uniform(distance_down, distance_up)
    x_shift = random.uniform(distance_down, distance_up)

    # 生成平移矩阵
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    # 平移
    img_shift = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

    return img_shift


def crop(img, crop_x, crop_y):
    '''
    读取部分图像，进行裁剪
    :param img:
    :param crop_x:裁剪x尺寸
    :param crop_y:裁剪y尺寸
    :return:
    '''
    rows, cols = img.shape[:2]
    # 偏移像素点
    x_offset = random.randint(0, cols - crop_x)
    y_offset = random.randint(0, rows - crop_y)

    # 读取部分图像
    img_part = img[y_offset:(y_offset+crop_y), x_offset:(x_offset+crop_x)]

    return img_part


def lighting_adjust(img, k_down, k_up, b_down, b_up):
    '''
    图像亮度、对比度调整
    :param img:
    :param k_down:对比度系数下限
    :param k_up:对比度系数上限
    :param b_down:亮度增值上限
    :param b_up:亮度增值下限
    :return:调整后的图像
    '''
    # 对比度调整系数
    slope = random.uniform(k_down, k_up)
    # 亮度调整系数
    bias = random.uniform(b_down, b_up)
    # 图像亮度和对比度调整
    img = img * slope + bias
    # 灰度值截断，防止超出255
    img = np.clip(img, 0, 255)

    return img.astype(np.uint8)


def Gaussian_noise(img, mean=0, std=1):
    '''
    图像加高斯噪声
    :param img: 原图
    :param mean: 均值
    :param std: 标准差
    :return:
    '''
    # 高斯噪声图像
    gauss = np.random.normal(loc=mean, scale=std, size=img.shape)
    img_gauss = img + gauss

    # 裁剪
    out = np.clip(img_gauss, 0, 255)

    return out


def normalization(img, mean, std):
    '''
    图像归一化,图像像素点从(0,255)->（0,1）
    :param img:
    :param mean:所有样本图像均值
    :param std: 所有样本图像标准差
    :return:
    '''
    img -= mean
    img /= std

    return img
