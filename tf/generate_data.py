# _*_ coding:utf-8 _*_
# Created time: 4/4 2018
# Author : JosiahMg
# env : anaconda python 3.5

import numpy as np
import matplotlib.pyplot as plt

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

