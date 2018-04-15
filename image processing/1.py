# coding = utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt

input_path = '/media/liuzhicai/资料/原始图像/原始图像/'
output_path = '/media/liuzhicai/资料/原始图像/原始图像/output/'

############### 颜色空间转换################

image = cv2.imread(input_path + 'image0011.tif')

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

'''
plt.figure(figsize=(8, 2))  # 这个比例关系到画出来的图像边缘
plt.subplot(131)
plt.xticks([]), plt.yticks([])
plt.imshow(image_rgb)
plt.title('original')
plt.subplot(132)
plt.xticks([]), plt.yticks([])
plt.imshow(image_gray, cmap = 'gray')
plt.title('gray')
plt.subplot(133)
plt.xticks([]), plt.yticks([])
plt.imshow(image_hsv)
plt.title('hsv')
plt.tight_layout()        # 调整图像边缘
# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.1, wspace=0.1)
plt.savefig(output_path + 'color_shift.jpg')
plt.show()
'''

'''
##############阈值分割################

# 固定阈值法
ret1, th1 = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY)

# Otsu阈值法
ret2, th2 = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 先进行高斯滤波，再使用Otsu阈值法
blur = cv2.GaussianBlur(image_gray, (5, 5), 0)  # 参数3是模糊系数，越大越明显，0代表默认值0.8
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

images = [image_gray, 0, th1, image_gray, 0, th2, blur, 0, th3]
titles = ['Original', 'Histogram', 'Global(v=100)',
          'Original', 'Histogram', "Otsu's",
          'Gaussian filtered Image', 'Histogram', "Otsu's"]

# plt.figure(figsize=(5, 5))
for i in range(3):
    # 绘制原图
    plt.subplot(3, 3, i * 3 + 1)
    plt.imshow(images[i * 3], 'gray')
    plt.title(titles[i * 3], fontsize=8)
    plt.xticks([]), plt.yticks([])

    # 绘制直方图plt.hist，ravel函数将数组降成一维
    plt.subplot(3, 3, i * 3 + 2)
    plt.hist(images[i * 3].ravel(), 256)
    plt.title(titles[i * 3 + 1], fontsize=8)
    plt.xticks([]), plt.yticks([])

    # 绘制阈值图
    plt.subplot(3, 3, i * 3 + 3)
    plt.imshow(images[i * 3 + 2], 'gray')
    plt.title(titles[i * 3 + 2], fontsize=8)
    plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.savefig(output_path+'binary.jpg')
plt.show()
'''

'''
###############几何变换#############

# 缩放
image_resize = cv2.resize(image, (640, 480))
# image_resize = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)
# 平移
rows, cols = image.shape[:2]
M = np.float32([[1, 0, 800], [0, 1, 600]]) #x轴平移100，y轴平移50
dst = cv2.warpAffine(image, M, (cols, rows)) # 第3个参数为变换后的图像大小
# 旋转
M1 = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)   # 旋转中心，角度，缩放比例
dst1 = cv2.warpAffine(image, M1, (cols, rows))
# 翻转图像
dst2 = cv2.flip(image, 1)    # 参数2=0：垂直翻转，>0：水平翻转，<0：水平垂直翻转

image_list = [image, image_resize, dst, dst1, dst2]
title_list = ['original', 'resize', 'translation', 'rotation', 'flip']

plt.figure(figsize=(7,1))
for i in range(5):
    plt.subplot(1, 5, i+1)
    image_list[i] = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2RGB)
    plt.imshow(image_list[i])
    plt.title(title_list[i])
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.savefig(output_path + 'geometry_shift.jpg')
plt.show()
'''

'''
###############平滑图像#################

blur = cv2.blur(image, (5, 5))
gaussian = cv2.GaussianBlur(image, (5, 5), 1)
median = cv2.medianBlur(image, 5)
bilateral = cv2.bilateralFilter(image, 9, 75, 75)

total = np.hstack((image, blur, gaussian, median, bilateral))
cv2.imwrite(output_path+'smoothing.jpg', total)
cv2.namedWindow('total', cv2.WINDOW_NORMAL)
cv2.imshow('total', total)
cv2.waitKey(0)
'''

'''
###############canny边缘检测#############

edges = cv2.Canny(image_gray, 30, 70)

# 先进行阈值分割，后边缘检测
_, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edges1 = cv2.Canny(thresh, 30, 70)

total = np.hstack((image_gray, edges, edges1))

cv2.imwrite(output_path+'canny.jpg', total)
cv2.namedWindow('canny', cv2.WINDOW_NORMAL)
cv2.imshow('canny', total)
cv2.waitKey(0)
'''

###############轮廓提取##################

# 先进行中值滤波
# image_gray = cv2.medianBlur(image_gray, 21)
# 先进行双边滤波
image_gray = cv2.bilateralFilter(image_gray, 17, 75, 75)
# OTSU图像二值化
ret, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)   # cv2.THRESH_BINARY 会在外面画一个框
img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))

cv2.drawContours(image, contours, -1, (0, 0, 255), 10)   # -1表示画出所有边缘
cv2.imwrite(output_path+'outlines_smoothing.jpg', image)
cv2.namedWindow('contours', cv2.WINDOW_NORMAL)
cv2.imshow('contours', image)
cv2.waitKey(0)



