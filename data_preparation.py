# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 22:53:58 2019

@author: wangz
"""
import tensorflow as tf
import cv2
import numpy as np
import imageio
import os
import numpy
import random
from skimage import io,transform,color,data
from skimage import exposure
import pywt
import matplotlib.pyplot as plt
#定义一个函数将gif格式转化为矩阵形式
def readImg(im_fn):    
    im = cv2.imread(im_fn)
    if im is None :
        #print('{} cv2.imread failed'.format(im_fn))
        tmp = imageio.mimread(im_fn)
        if tmp is not None:
            imt = np.array(tmp)
            imt = imt[0]
            im = imt[:,:,0:3]
            im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)#将bgr转化成rgb格式
    return im

def _image_noise(images, mean=0, std=0.01):
        # 图像噪声
        old_image = images
        new_image = old_image
        for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                    new_image[i, j] += random.gauss(mean, std)#高斯噪声
        images = new_image
        
        return images
##白化
def image_whitening(images):
        # 图像白化
    old_image = images
    new_image = (old_image - np.mean(old_image)) / np.std(old_image)
    images = new_image       
    return images


def normalization_image(imagedata):
    # normalize data to 0~1 for every element of 2-D array 'imagedata'
    maxvalue = imagedata.max()
    minvalue = imagedata.min()
    data = np.multiply((imagedata - minvalue), 1.0/(maxvalue - minvalue))
    print ('Succesfully normalize image')
    return data


def channel_denoising_image(imagedata):#通道归一化
    # eliminate the channel effect
    # 'imagedata':2-D array
    # use method:f = g-rowmean + globlemean
    # g indicates 'imagedata','rowmean'indicates mean of each row in 'imagedata'
    # 'globlemean'indicates mean of whole 2-D array 'imagedata'
    meanvalue = imagedata.mean()
    channelmeanvalue = imagedata.mean(1)
    i=0
    while i<100:
        imagedata[i] = (imagedata[i]-channelmeanvalue[i] + meanvalue)

        i = i+1
    print ('Succesfully eliminate the channel effect')
    return imagedata


def resize(image):
    image=cv2.resize(image,(1700,600),interpolation=cv2.INTER_CUBIC)
    return image

'''
灰度拉伸
定义：灰度拉伸，也称对比度拉伸，是一种简单的线性点运算。作用：扩展图像的
      直方图，使其充满整个灰度等级范围内
公式：
g(x,y) = 255 / (B - A) * [f(x,y) - A],
其中，A = min[f(x,y)],最小灰度级；B = max[f(x,y)],最大灰度级；
     f(x,y)为输入图像,g(x,y)为输出图像
缺点：如果灰度图像中最小值A=0，最大值B=255，则图像没有什么改变
'''
def grey_scale(image):
    #img_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)    
    rows,cols = image.shape
    flat_gray = image.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    print('A = %d,B = %d' %(A,B))
    output = np.uint8(255 / (B - A) * (image - A) + 0.5)
    return output


#伽玛变换
def gamma(img, c, v):
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = c * i ** v
    output_img = cv2.LUT(img, lut) #像素灰度值的映射
    output_img = np.uint8(output_img+0.5)  
    return output_img


def convert_gray(f):
     rgb=io.imread(f)    #依次读取rgb图片 
     image=resize(rgb)
     #图像灰度伽玛变换
     image = gamma(image, 0.00000005, 4.0)
     gray=color.rgb2gray(image)   #将rgb图片转换成灰度图
    
     dst = gray[100:450, 250:1450]  # 裁剪坐标为[y0:y1, x0:x1]
     #dst = rgb[200:200, 0:1200]   
     dst= exposure.adjust_gamma(dst,0.5)#    对比度
     dst = channel_denoising_image(dst)#通道归一化
     dst = _image_noise(dst)#去噪
     dst=grey_scale(dst)#灰度拉伸
     data = normalization_image(dst) #归一化
     return data


def wave(image):
    #img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs
    return cV

str='preparation\\1\\0\\*.jpg' #原始0型数据所在的文件夹
coll = io.ImageCollection(str,load_func=convert_gray)
for i in range(len(coll)):
    io.imsave('preparation\\19_1\\0\\'+np.str(i)+'.jpg',coll[i])  #循环保存图片,新数据所在的路径

str='preparation\\1\\1\\*.jpg' 
coll = io.ImageCollection(str,load_func=convert_gray)
for i in range(len(coll)):
    io.imsave('preparation\\19_1\\1\\'+np.str(i)+'.jpg',coll[i])  #循环保存图片
      
str='preparation\\1\\3\\*.jpg' 
coll = io.ImageCollection(str,load_func=convert_gray)
for i in range(len(coll)):
    io.imsave('preparation\\19_1\\3\\'+np.str(i)+'.jpg',coll[i])  #循环保存图片
 

####################        进一步实现  ####################


def resize1(image):    
    M = cv2.imread("Preparation\\19_1\\0\\73.jpg")#从处理过的图像中选取一张背景图片
    subtracted = cv2.subtract(image,M)#将图像im与M相减
    #image=cv2.resize(subtracted,(300,150),interpolation=cv2.INTER_CUBIC)
    return subtracted
    #return image
    
def convert_gray_1(f): 
     rgb=io.imread(f)    #依次读取rgb图片
     rgb=resize2(rgb)
     
     #rgb=wave(rgb)
     #return rgb
     #rgb = cv2.medianBlur(rgb,3) #使用3*3的中值滤波器滤除椒盐噪声
     #imgGlassesGray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
     image = cv2.equalizeHist(rgb)#直方图均衡化
     image= exposure.adjust_gamma(rgb,0.5)#    对比度  
     ret, image = cv2.threshold(image, 200, 255,cv2.THRESH_BINARY)   #二值化
     image = cv2.medianBlur(image,3)        #使用3*3的中值滤波器滤除椒盐噪声
     #data = normalization_image(image)   #归一化
     return image 


str='preparation\\19_1\\0\\*.jpg' #所要处理的原始图像文件夹地址
coll = io.ImageCollection(str,load_func=convert_gray_1)
for i in range(len(coll)):
    io.imsave('preparation\\19_2\\0\\'+np.str(i)+'.jpg',coll[i])  #循环保存图片所在文件夹

str='preparation\\19_1\\1\\*.jpg' 
coll = io.ImageCollection(str,load_func=convert_gray_1)
for i in range(len(coll)):
    io.imsave('preparation\\19_2\\1\\'+np.str(i)+'.jpg',coll[i])  #循环保存图片

str='preparation\\19_1\\3\\*.jpg' 
coll = io.ImageCollection(str,load_func=convert_gray_1)
for i in range(len(coll)):
    io.imsave('preparation\\19_2\\3\\'+np.str(i)+'.jpg',coll[i])  #循环保存图片

####################            ####################            

def convert_gray_3(f): 
    
     rgb=io.imread(f)    #依次读取rgb图片
     #imgGlassesGray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
     image = cv2.equalizeHist(rgb)
     image= exposure.adjust_gamma(image,15)#    对比度
     gray_src = cv2.bitwise_not(image)    #二值化
     #应用一个自适应阈值，根据公式将灰度图像转换为二进制图像
     #binary_src = cv2.adaptiveThreshold(gray_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)  
     #vline = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(100 / 2)), (-1, -1)) #定义一个结构元素
     #dst = cv2.morphologyEx(binary_src, cv2.MORPH_OPEN, vline)  #开运算
     dst = cv2.medianBlur(gray_src ,3)        #   使用3*3的中值滤波器滤除椒盐噪声 
     data3 = normalization_image(dst)#     归一化
     return data3

str='preparation\\19_1\\3\\*.jpg' 
coll = io.ImageCollection(str,load_func=convert_gray_3)
for i in range(len(coll)):
    io.imsave('preparation\\20_1\\3\\'+np.str(i)+'.jpg',coll[i])  #循环保存图片 
 

def resize2(image):
    image=cv2.resize(image,(450,150),interpolation=cv2.INTER_CUBIC)
    return image
def convert(f): 
     rgb=io.imread(f)    #依次读取rgb图片
     rgb=resize2(rgb)
     return rgb

str='preparation\\19_1\\0\\*.jpg' 
coll = io.ImageCollection(str,load_func=convert)
for i in range(len(coll)):
    io.imsave('preparation\\picture\\0\\'+np.str(i)+'.jpg',coll[i])  #循环保存图片

str='preparation\\19_1\\1\\*.jpg' 
coll = io.ImageCollection(str,load_func=convert)
for i in range(len(coll)):
    io.imsave('preparation\\picture\\1\\'+np.str(i)+'.jpg',coll[i])  #循环保存图片

str='preparation\\19_1\\3\\*.jpg' 
coll = io.ImageCollection(str,load_func=convert)
for i in range(len(coll)):
    io.imsave('preparation\\picture\\3\\'+np.str(i)+'.jpg',coll[i])  #循环保存图片    


