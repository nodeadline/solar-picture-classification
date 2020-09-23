# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:10:01 2019

@author: wangz
"""

#-*- coding: UTF-8 -*-   
 
from PIL import Image
from PIL import ImageEnhance
import cv2
import os
import random
'''
def augument(image_path, parent):
     #读取图片
     image = Image.open(image_path)
 
     image_name = os.path.split(image_path)[1]
     name = os.path.splitext(image_name)[0]

     #变亮
     #亮度增强,增强因子为0.0将产生黑色图像；为1.0将保持原始图像。
     enh_bri = ImageEnhance.Brightness(image)
     brightness = 1.3
     image_brightened1 = enh_bri.enhance(brightness)
     image_brightened1.save(os.path.join(parent, '{}_bri1.jpg'.format(name)))
 
     #变暗
     enh_bri = ImageEnhance.Brightness(image)
     brightness = 0.8
     image_brightened2 = enh_bri.enhance(brightness)
     image_brightened2.save(os.path.join(parent, '{}_bri2.jpg'.format(name)))
 
     #色度,增强因子为1.0是原始图像
     # 色度增强
     enh_col = ImageEnhance.Color(image)
     color = 2#200
     image_colored1 = enh_col.enhance(color)
     image_colored1.save(os.path.join(parent, '{}_col1.jpg'.format(name)))

     # 色度减弱
     enh_col = ImageEnhance.Color(image)
     color = 0.8
     image_colored1 = enh_col.enhance(color)
     image_colored1.save(os.path.join(parent, '{}_col2.jpg'.format(name)))
 
     #对比度，增强因子为1.0是原始图片
     # 对比度增强
     enh_con = ImageEnhance.Contrast(image)
     contrast = 2#2.5
     image_contrasted1 = enh_con.enhance(contrast)
     image_contrasted1.save(os.path.join(parent, '{}_con1.jpg'.format(name)))
 
     # 对比度减弱
     enh_con = ImageEnhance.Contrast(image)
     contrast = 0.8
     image_contrasted2 = enh_con.enhance(contrast)
     image_contrasted2.save(os.path.join(parent, '{}_con2.jpg'.format(name)))
   
     # 锐度，增强因子为1.0是原始图片,锐度改变效果不好
     # 锐度增强
     enh_sha = ImageEnhance.Sharpness(image)
     sharpness = 3.0
     image_sharped1 = enh_sha.enhance(sharpness)
     image_sharped1.save(os.path.join(parent, '{}_sha1.jpg'.format(name)))
 
     # 锐度减弱
     enh_sha = ImageEnhance.Sharpness(image)
     sharpness = 0.8
     image_sharped2 = enh_sha.enhance(sharpness)
     image_sharped2.save(os.path.join(parent, '{}_sha2.jpg'.format(name)))
    
'''
###############3自定义图像增强手段，可以自己选择合适的因子
def augument(image_path, parent):
     #读取图片
    image = Image.open(image_path)
 
    image_name = os.path.split(image_path)[1]
    name = os.path.splitext(image_name)[0]
    '''     
     #色度,增强因子为1.0是原始图像
     # 色度增强
     enh_col = ImageEnhance.Color(image)
     color = 20#200
     image_colored1 = enh_col.enhance(color)
     image_colored1.save(os.path.join(parent, '{}_col1.jpg'.format(name)))
    '''
    '''     
     #变亮
     #亮度增强,增强因子为0.0将产生黑色图像；为1.0将保持原始图像。
     enh_bri = ImageEnhance.Brightness(image)
     brightness = 1.2
     image_brightened1 = enh_bri.enhance(brightness)
     image_brightened1.save(os.path.join(parent, '{}_bri1.jpg'.format(name)))
    '''    
    #对比度，增强因子为1.0是原始图片
    # 对比度增强
    enh_con = ImageEnhance.Contrast(image)
    contrast = 2.5#5
    image_contrasted1 = enh_con.enhance(contrast)
    image_contrasted1.save(os.path.join(parent, '{}_col1.jpg'.format(name)))
    '''
     # 锐度减弱
    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = 0.8
    image_sharped2 = enh_sha.enhance(sharpness)
    image_sharped2.save(os.path.join(parent, '{}_sha2.jpg'.format(name)))
    '''



dir = 'preparation\\19_3\\0\\'##########所要增强图片所在的文件夹
for parent, dirnames, filenames in os.walk(dir):
     for filename in filenames:
          fullpath = os.path.join(parent + '/', filename)
          if 'jpg' in fullpath:
               print(fullpath, parent)
               augument(fullpath, parent)



######################################加入高斯噪声
def _image_noise(images, mean=0, std=0.01):
        # 图像噪声
        old_image = images
        new_image = old_image
        for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                    new_image[i, j] += random.gauss(mean, std)
        images = new_image
        
        return images

# encoding:utf-8
######################图像翻转，加入随机噪声
def enhance(image_path, parent):
         
    image = cv2.imread(image_path)#读取图片
    image_name = os.path.split(image_path)[1]
    name = os.path.splitext(image_name)[0]
    ''' 
    # Flipped Horizontally 水平翻转
    h_flip = cv2.flip(image, 1)
    h_flip.save(os.path.join(parent, '{}_col1.jpg'.format(name)))
    #cv2.imencode('.jpg', image)[1].tofile(img_path_change)
    # Flipped Vertically 垂直翻转
    v_flip = cv2.flip(image, 0)
    v_flip.save(os.path.join(parent, '{}_col2.jpg'.format(name)))
    '''
    # Flipped Horizontally & Vertically 水平垂直翻转
    hv_flip = cv2.flip(image, -1)
    hv_flip.save(os.path.join(parent, '{}_col3.jpg'.format(name)))
    '''
    image_noise=_image_noise(image)
    image_noise.save(os.path.join(parent, '{}_col4.jpg'.format(name)))
    '''

dir = 'preparation\\19_2\\0\\'##########所要增强图片所在的文件夹
for parent, dirnames, filenames in os.walk(dir):
     for filename in filenames:
          fullpath = os.path.join(parent + '/', filename)
          if 'jpg' in fullpath:
               print(fullpath, parent)
               enhance(fullpath, parent)
               
















