# _*_ coding:utf-8 _*_

# Created time : 4/11 2018
# Author : JosiahMg
# Env: anaconda python 3.6

import numpy as np
import operator


def create_data_set():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    lables = ['a', 'a', 'b', 'b']
    return group, lables


grp, labels = create_data_set()
print(grp, labels)



