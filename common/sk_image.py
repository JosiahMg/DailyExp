# _*_ coding:utf-8 _*_
# Created time : 3/16 2018
# Author: JosiahMg
# env: anaconda python3.6


import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io, img_as_ubyte, img_as_float
import skimage
from skimage import exposure


# 产生随机数生成图片 300 *300
def show_random_image():
    random_image = np.random.random([300,300])
    plt.imshow(random_image, cmap='gray')
    plt.colorbar()
    plt.show()


# 显示 硬币图片
def show_coins_image():
    coins = data.coins()
    plt.imshow(coins, cmap='gray')
    plt.show()


# 显示彩色的猫图片
def show_cat_image():
    cat = data.chelsea()
    plt.imshow(cat)
    plt.colorbar()
    plt.show()


# 0-1 和 0-255 数据显示相同的图片
def diff_data_show_image():
    linear0 = np.linspace(0, 1, 2500).reshape((50, 50))
    linear1 = np.linspace(0, 255, 2500).reshape((50, 50)).astype(np.uint8)
    fig, (ax0, ax1) = plt.subplots(1,2)

    ax0.imshow(linear0, cmap='gray')
    ax1.imshow(linear1, cmap='gray')
    plt.show()

# 0-1 和 0-255 数据转换
def float_and_ubyte():
    img = data.chelsea()
    image_float = img_as_float(img)
    image_ubyte = img_as_ubyte(img)
    print('type min max:',image_float.dtype, image_float.min(), image_float.max())
    print('type min max:',image_ubyte.dtype, image_ubyte.min(), image_ubyte.max())


# 图像IO
# io.imread()  读取单个图片
# io.imread_collection()  读取多个图片

def read_single_img():
    img = io.imread('../source/cv.jpg')
    plt.imshow(img)
    plt.show()


# 显示多个图片

def read_mul_img():
    ic = io.imread_collection('../source/*.jpg')
    f, axes = plt.subplots(nrows=1, ncols=len(ic), figsize=(15, 10))
    for i, image in enumerate(ic):
        axes[i].imshow(image)
        axes[i].axis('off')
    plt.show()


# 保持图片

def save_img():
    save_img = io.imread('../source/cv.jpg')
    io.imsave('../source/cv_save.png', save_img)


# RGB->GRAY
def color_gray():
    color_image = data.chelsea()
    gray_img = skimage.color.rgb2gray(color_image)
    plt.imshow(gray_img, cmap='gray')
    plt.show()


# 颜色直方图

def show_histgram():
    image = data.camera()
    # bin_centers = [0 , 255]
    # hist :each of point values
    hist, bin_centers = exposure.histogram(image)
    plt.fill_between(bin_centers, hist)
    plt.ylim(0)
    plt.show()


#显示不同颜色通道的柱状图

def show_channels_histogram():
    cat = data.chelsea()
    hist_r, bin_centers_r = exposure.histogram(cat[:, :, 0])
    hist_g, bin_centers_g = exposure.histogram(cat[:, :, 1])
    hist_b, bin_centers_b = exposure.histogram(cat[:, :, 2])
    plt.figure(figsize=(10, 5))

    ax = plt.subplot(131)
    plt.fill_between(bin_centers_r, hist_r, facecolor='r')
    plt.ylim(0)

    plt.subplot(132, sharey=ax)
    plt.fill_between(bin_centers_g, hist_g, facecolor='g')
    plt.ylim(0)

    plt.subplot(133, sharey=ax)
    plt.fill_between(bin_centers_b, hist_b, facecolor='b')
    plt.ylim(0)

    plt.show()


# 对比度
# 将小于某像素的值改成0  将大于某像素的点值改成255
# exposure.rescale_intensity
def show_img_contrast():
    img = data.camera()
    hist, bin_centers = exposure.histogram(img)
    # 将<10 像素的点改成0  将>180的像素点改成255
    high_contrast = exposure.rescale_intensity(img, in_range=(10, 180))
    hist2, bin_centers2 = exposure.histogram(high_contrast)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,5))
    ax1.imshow(img, cmap='gray')
    ax2.imshow(high_contrast, cmap='gray')

    fig, (ax_hist1, ax_hist2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax_hist1.fill_between(bin_centers, hist)
    ax_hist2.fill_between(bin_centers2, hist2)
    plt.ylim(0)
    plt.show()


# 直方图均衡化
# 将像素点的值平均分布，使得细节更明细
# exposure.equalize_hist
def show_img_equalize():
    img = data.camera()
    hist, bin_centers = exposure.histogram(img)

    equalized = exposure.equalize_hist(img)
    hist2, bin_centers2 = exposure.histogram(equalized)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.imshow(img, cmap='gray')
    ax2.imshow(equalized, cmap='gray')

    fig, (ax_hist1, ax_hist2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax_hist1.fill_between(bin_centers, hist)
    ax_hist2.fill_between(bin_centers2, hist2)
    plt.ylim(0)
    plt.show()


# 中值滤波

from skimage.morphology import disk
from skimage.filters.rank import median


def show_img_median():
    img = data.camera()
    # 3*3  and 5*5  滤波器
    med1 = median(img, disk(3))
    med2 = median(img, disk(5))

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,10))
    ax1.imshow(img, cmap='gray')
    ax2.imshow(med1, cmap='gray')
    ax3.imshow(med2, cmap='gray')
    plt.show()


# 高斯滤波

from skimage.filters import gaussian


def show_img_gaussian():
    img = data.camera()
    gas1 = gaussian(img, sigma=3)
    gas2 = gaussian(img, sigma=5)

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 10))
    ax1.imshow(img, cmap='gray')
    ax2.imshow(gas1, cmap='gray')
    ax3.imshow(gas2, cmap='gray')
    plt.show()


# 均值滤波
from skimage.filters.rank import mean


def show_img_mean():
    img = data.camera()
    mean1 = mean(img, disk(3))
    mean2 = mean(img, disk(5))
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 10))
    ax1.imshow(img, cmap='gray')
    ax2.imshow(mean1, cmap='gray')
    ax3.imshow(mean2, cmap='gray')
    plt.show()


# 边缘检测
# 原理:根据梯度算法计算梯度下降快的地方为边界

from skimage.filters import prewitt, sobel


def show_img_edge():
    img = data.camera()
    edge_prewitt = prewitt(img)
    edge_sobel = sobel(img)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    # Prewitt 边缘检测
    ax1.imshow(edge_prewitt, cmap=plt.cm.gray)
    ax2.imshow(edge_sobel, cmap=plt.cm.gray)
    plt.show()


# 图像特征
# 颜色特征

def color_feature():
    camera = img_as_float(data.camera())

    hist, bin_centers = exposure.histogram(camera, nbins=10)

    plt.fill_between(bin_centers, hist)
    plt.show()


from skimage.feature import daisy

# SIFT 特征(DAISY 特征)
# Scale-invariant feature transform


def daisy_feature():
    camera = data.camera()
    daisy_feat, daisy_img = daisy(camera, step=180, radius=58, rings=2, histograms=6, visualize=True)
    print(daisy_img.shape)
    plt.imshow(daisy_img)
    plt.show()


# HOG 特征
# 适合做图像的人体检测

from skimage.feature import hog


def hog_feature():
    camera = data.camera()
    hog_feat, hog_img = hog(camera, visualise=True)
    plt.imshow(hog_img)
    plt.show()


# K-Means 聚类及图像压缩

from sklearn.cluster import KMeans

# 将读取图片之后将之转换成(h*3, depth)维度的数据并进行聚类
# original_img 需要聚类的图片


def kmeans_fit_img(original_img, k=5):
    original_img = img_as_ubyte(original_img)
    height, width, depth = original_img.shape
    pixel_sample = np.reshape(original_img, (height*width, depth))
    kmeans = KMeans(n_clusters=k, random_state=0)

    kmeans.fit(pixel_sample)
    # 返回每个像素点属于哪个类别 len(cluster_assignments) = height*width
    cluster_assignments = kmeans.predict(pixel_sample)

    # set is {0, 1, 2, 3, 4}  即k=5
    print(set(cluster_assignments))

    # cluster_centers 返回聚类点的像素值 共有5个,即shape=5*3
    cluster_centers = kmeans.cluster_centers_
    print(cluster_centers.shape)
    print(cluster_centers)
    return cluster_assignments, cluster_centers


# 将图像压缩 将相似的像素点使用同一个像素值替代
# img ：需要压缩的图片
# cluster_assignments: 聚类返回的像素索引
# cluster_centers: 聚类中心像素值
def image_compress(img, cluster_assignments, cluster_centers):
    original_img = img_as_ubyte(img)
    height, width, depth = img.shape

    compressed_img = np.zeros((height, width, depth), dtype=np.uint8)
    pixel_count = 0
    # 遍历每个像素点
    for i in range(height):
        for j in range(width):
            # 获取每个像素点的中心索引
            cluster_idx = cluster_assignments[pixel_count]

            #根据索引值获取聚类的像素点
            cluster_value = cluster_centers[cluster_idx]
            # 将像素点放置到新的图像区域
            compressed_img[i][j] = cluster_value
            pixel_count += 1

    io.imsave('../source/compress.jpg', compressed_img)

    plt.subplot(121), plt.title('Original Image'), plt.imshow(img), plt.axis('off')
    plt.subplot(122), plt.title('Compressed Image'), plt.imshow(compressed_img), plt.axis('off')
    plt.show()

# 使用kmeans进行压缩的测试程序


def kmeans_compress_test():
    img = io.imread('../source/yt.jpg')
    assigments, centers = kmeans_fit_img(img, k=100)
    image_compress(img, assigments, centers)



