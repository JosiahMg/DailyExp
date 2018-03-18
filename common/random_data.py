# _*_ coding:utf-8 _*_

import numpy as np

"""
汇总产生随机数的函数
1、产生[0, 1)的数据
random.random(size=None)
random.random_sample(size=None)
random.sample(size=None)
random.random.rand(d0, d1, ..., dn)

2、产生整数
random.randint(low, high=None, size=None, dtype='l')

3、产生标准正态分布的数据
random.randn(d0, d1, ..., dn)
"""


# 产生一个随机数组[0, 1)
data = np.random.random([500, 500])
data = np.random.random_sample([2, 34])
data = np.random.sample([4, 2, 3])
print(data)


# np.random.rand(d0, d1, d2, ..., dn)
# rand 根据维度生成[0, 1)之间的数据

"""
4 rows * 2 cols
[[ 0.44699534  0.19614608]
 [ 0.97601286  0.62321998]
 [ 0.95397992  0.89037453]
 [ 0.24367989  0.10343284]]
"""
data = np.random.rand(4, 2, 4)


"""
1 rows * 4 cols
the result is :
[[ 0.43276225  0.10048263  0.72269826  0.93407941]]

"""
data = np.random.rand(1,4)



"""
the following two equations are equivalent.
the result is :
[ 0.63598041  0.16792179  0.45834605  0.3686105 ]
"""
data = np.random.rand(4)
data = np.random.rand(4, )



# np.random.randn() 具有标正态分布
# np.random.randn(d0, d1, ..., dn)
# standard normal distribution

data = np.random.randn(1, 5)

# np.random.randint()
# np.random.randint(low, high=None, size=None, dtype='l')
# 返回随机整数，范围[low, high)  size为维度大小
# high如果不写则默认生成的随机数范围 [0, low)

data = np.random.randint(5,10,size=(3,5))
