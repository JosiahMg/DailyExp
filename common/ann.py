# _*_coding:utf-8 _*_
# Created time:3/30 2018
# Author: JosiahMg


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


fruits_df = pd.read_table('../source/fruit_data_with_colors.txt')

'''
fruits_df[['width', 'height']] 返回的id是新的地址空间
fruits_df['fruit_label']  返回的id是原先fruits_df的地址空间，所以需要copy()


'''
X = fruits_df[['width', 'height']]
y = fruits_df['fruit_label'].copy()

# 将y非1的情况改成0，因为数据量比较小，所以只考虑苹果和非苹果两种情况
y[y != 1] = 0


# ANN 数据准备
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)


from sklearn.neural_network import MLPClassifier
from ml_visualization import plot_class_regions_for_classifier

# 单层ANN

units = [1, 10, 100]

'''
hidden_layer_sizes=[5,2,5] 列表中填写每个隐藏层的个数
上面的表示：第一层隐藏层5个节点，第二层2个节点，第三层5个节点
'''


def single_ann(units):
    for unit in units:
        ann_model = MLPClassifier(hidden_layer_sizes=[unit], activation='relu', solver='lbfgs', random_state=0)
        ann_model.fit(X_train, y_train)
        print('神经元的个数={} 准确率{:.3f}'.format(unit, ann_model.score(X_test, y_test)))
        plot_class_regions_for_classifier(ann_model, X_test.values, y_test.values, title='Unit={}'.format(unit))


# 多层ANN


def mul_ann():
    ann_model = MLPClassifier(hidden_layer_sizes=[10, 10], activation='relu', solver='lbfgs', random_state=0)
    ann_model.fit(X_train, y_train)
    print('准确率:{:.3f}'.format(ann_model.score(X_test, y_test)))
    plot_class_regions_for_classifier(ann_model, X_test.values,y_test.values)


'''
Regular Expression
正则化的值越小说明约束越小，模型产生的预测能力越差，相当抛硬币
正则化的值越大则会产生过拟合
'''
aplhas = [0.0001, 0.01, 0.1, 1.0]


def regular_ann(aplhas):
    for alpha in aplhas:
        ann_model = MLPClassifier(hidden_layer_sizes=[100, 100], activation='tanh', solver='lbfgs', random_state=0,alpha=alpha)
        ann_model.fit(X_train, y_train)
        print('alpha={},准确率={:.3f}'.format(alpha, ann_model.score(X_test, y_test)))
        plot_class_regions_for_classifier(ann_model, X_test.values, y_test.values, title='alpha={}'.format(alpha))


regular_ann(aplhas)