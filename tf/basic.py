# _*_ coding:utf-8 _*_
# Created time : 4/5 2018
# Author : JosiahMg
# env : anaconda python 3.5


import tensorflow as tf

# 计算图
a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([1.0, 2.0], name='b')
result = tf.add(a, b, name='add')

# 会话 执行运算结果 三种使用方法

# 方式1 直接使用
sess = tf.Session()
sess.run(result)
sess.close()


# 方式2 通过python上下文管理器来管理会话
with tf.Session() as sess:
    sess.run(result)


# 方式3 通过python上下文管理器来管理会话
sess = tf.Session()
with sess.as_default():
    result.eval()


# placeholder 占位，在run时制定值，传递真实的样本
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
with tf.Session() as sess:
    sess.run(adder_node, feed_dict={a: 3, b: 4.5})
    sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]})

# Variable 可变的值，在训练过程中可以被修改
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(linear_model, feed_dict={x: [1, 2, 3, 4]})

# tf.assign  赋值
W = tf.assign(W, [-1.])




