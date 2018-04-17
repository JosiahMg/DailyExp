# _*_ coding:utf-8 _*_
# Created time: 4/4 2018
# Author : JosiahMg
# env : anaconda python 3.5

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

seed = 2

# 产生数据


def generate_data():
    rdm = np.random.RandomState(seed)
    X = rdm.randn(300, 2)
    Y_ = [int(x0*x0 + x1*x1 < 2) for (x0, x1) in X]
    Y_c = [['red' if y else 'blue'] for y in Y_]
    X = np.vstack(X).reshape(-1, 2)
    Y_ = np.vstack(Y_).reshape(-1, 1)

    return X, Y_, Y_c


# 读取source/handwrite_pic下的图片并转换成(0-1)之间的数据值
# 处理的结果是使图片的矩阵值和tensorflow中的input_data数据类似。
# pic_name:传入手写图片   return img_ready是返回1*784数据集


def pre_pic(pic_name):
    img = Image.open(pic_name)
    img_mat = img.resize((28, 28), Image.ANTIALIAS)
    # 转换为灰度图
    im_arr = np.array(img_mat.convert('L'))
    threshold = 50
    for i in range(28):
        for j in range(28):
            # 反色，实现黑底白字
            im_arr[i][j] = 255 - im_arr[i][j]
            if im_arr[i][j] < threshold:
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0/255.0)

    return img_ready



