# _*_ coding:utf-8 _*_

import tensorflow as tf

INPUT_NODE = 784
OUT_PUT = 10
LAYER1_NODE = 500

# 定义前向传播

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None: