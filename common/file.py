# _*_ coding:utf-8 _*_
# Created time: 3/30 2018
# Author : JosiahMg
# env: anaconda python3.6

# pickle 模块

import pickle


# 保持data 到path目录下
def pickle_save(data, path):
    save_file = open(path, 'wb')
    pickle.dump(data, save_file)
    save_file.close()


# 读取path目录下的数据
def pickle_load(path):
    load_file = open(path, 'rb')
    load_data = pickle.load(load_file)
    load_file.close()
    return load_data

