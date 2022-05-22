###
1、使用tensorflow3.6版本
```c++
      所需第三方库：
	tensorflow==1.10.0
	keras
	skimage
	pywt
	cv2
	matplotlib
	imageio
	pickle
	sklearn
```
###
2、程序说明：

data_preparation.py为图像预处理文件，包含本次实验所有的预处理方法，可灵活选用，由于部分图像爆发细节与背景较为相近，在滤波去噪时会丧失细节，造成纯黑图片

如果要增加数据样本，还需要进行翻转，色彩扰动等数据增强手段

####

jupter//put//data中为处理过的数据集

row_data为原始数据集

enhance.py  为图像增强文件

CNN.py实现了CNN框架下的图像识别模型

CNN-LSTM.py实现了CNN+LSTM框架下的图像识别模型

CNN_capsule.py
Capsule_Keras.py实现了CNN+Capsule框架下的图像识别模型

###
3、
太阳爆发图片可以在澳大利亚天文台找到。
在官网https://www.sws.bom.gov.au/World_Data_Centre/1/9/6，查阅图片爆发类型时刻等信息。
下载相关图片可以通过winscp连接地址ftp://ftp-out.sws.bom.gov.au/wdc/wdc_spec/data/learmonth/images/。（好像有两个天文台数据可选）
（winscp使用界面已分享）
![](https://zhiyang-blog.oss-cn-beijing.aliyuncs.com/image/2022-5-22-1.PNG)
![](https://zhiyang-blog.oss-cn-beijing.aliyuncs.com/image/2022-5-22-2.PNG)

当然也可以通过我的天翼云盘链接~
自制数据集只有1型，2型，3型爆发图片：
https://cloud.189.cn/t/NVv6JbRbYzEf (访问码:q9eg)





