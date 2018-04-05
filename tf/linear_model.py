# _*_ coding:utf-8 _*_
# Created time : utf-8
# Author: JosiahMg
# env: anaconda python 3.5

import tensorflow as tf
import numpy as np

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)

loss = tf.reduce_sum(tf.square(linear_model-y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(1000):
        sess.run(train, feed_dict={x: x_train, y: y_train})
        curr_w, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
        print('w is {} b is {} loss is {}'.format(curr_w, curr_b, curr_loss))



