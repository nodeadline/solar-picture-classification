# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 09:35:24 2019

@author: wangz
"""


import cv2
import numpy as np
from skimage import io
#import keras.backend as K
from keras.utils import np_utils, conv_utils
from keras.backend.common import normalize_data_format
from keras.models import Sequential
from keras.layers import Conv2D,Convolution2D, MaxPooling2D, Flatten, Dropout, Dense, Activation,LSTM,ZeroPadding2D,TimeDistributed,GlobalMaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2, l1
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc,confusion_matrix 
import os
import pickle
import keras
from keras.utils import plot_model
def get_name_list(filepath):                #获取各个类别的名字
    pathDir =  os.listdir(filepath)
    out = []
    for allDir in pathDir:
        if os.path.isdir(os.path.join(filepath,allDir)):
            child = allDir#.decode('gbk')    # .decode('gbk')是解决中文显示乱码问题
            out.append(child)
    return out
    
def eachFile(filepath):                 #将目录内的文件名放入列表中
    pathDir =  os.listdir(filepath)
    out = []
    for allDir in pathDir:
        child = allDir#.decode('gbk')    # .decode('gbk')是解决中文显示乱码问题
        out.append(child)
    return out


def get_2data(data_name,resize=True,data_format=None):   #当train和test数据被分为两个部分时使用  
    file_name = os.path.join(pic_dir_out,data_name+str(Width)+"X"+str(Height)+".pkl")   
    if os.path.exists(file_name):           #判断之前是否有存到文件中
        (X_train, y_train), (X_test, y_test) = pickle.load(open(file_name,"rb"))
        return (X_train, y_train), (X_test, y_test)   
    data_format = normalize_data_format(data_format)
    all_dir_set = eachFile(pic_dir_data)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
 
    for all_dir in all_dir_set:
        if not os.path.isdir(os.path.join(pic_dir_data,all_dir)):
            continue
        label = 0
        pic_dir_set = eachFile(os.path.join(pic_dir_data,all_dir))
        for pic_dir in pic_dir_set:
            print (pic_dir_data+pic_dir)
            if not os.path.isdir(os.path.join(pic_dir_data,all_dir,pic_dir)):
                continue    
            pic_set = eachFile(os.path.join(pic_dir_data,all_dir,pic_dir))
            for pic_name in pic_set:
                if not os.path.isfile(os.path.join(pic_dir_data,all_dir,pic_dir,pic_name)):
                    continue
                img = cv2.imread(os.path.join(pic_dir_data,all_dir,pic_dir,pic_name))
                if img is None:
                    continue
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
                if resize:
                    img = cv2.resize(img,(Width,Height))
                if (data_format == 'channels_last'):
                    img = img.reshape(-1,Width,Height,1)
                elif (data_format == 'channels_first'):
                    img = img.reshape(-1,1,Width,Height)
                if ('train' in all_dir):
                    X_train.append(img)
                    y_train.append(label)          
                elif ('test' in all_dir):
                    X_test.append(img)
                    y_test.append(label)
            if len(pic_set)!= 0:        
                label += 1
    X_train = np.concatenate(X_train,axis=0)        
    X_test = np.concatenate(X_test,axis=0)    
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    pickle.dump([(X_train, y_train), (X_test, y_test)],open(file_name,"wb")) 
    return (X_train, y_train), (X_test, y_test)

import matplotlib.pyplot  as plt
global Width, Height, pic_dir_out, pic_dir_data
Width = 300
Height = 150
pic_dir_out = 'jupter\\out\\'# 1994-98_2014_18_out 
pic_dir_data = 'jupter\\put\\data\\'#1994-98_2014_18_out/                  
def main():
    num_classes = 3  
    (X_train, y_train), (X_test, y_test) = get_2data("data_channel",resize=False,data_format='channels_last')

    print (X_train.shape)
    print (X_test.shape)

    ##########################################CNN+LSTM###############################
    X_train = X_train.reshape (-1,1,Width,Height,1)
    X_test = X_test.reshape (-1,1,Width,Height,1)
    #one-hot编码
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

   
    model = Sequential()                
    ##
    '''
    model.add(TimeDistributed(Conv2D(16, (3,3), activation= 'relu'),input_shape=(None,Width,Height,1)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(16, (3,3), activation= 'relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))  
    model.add(TimeDistributed(Conv2D(32, (3,3), activation= 'relu')))
    model.add(TimeDistributed(Conv2D(64, (3,3), activation= 'relu')))
    model.add(TimeDistributed(Conv2D(64, (3,3), activation= 'relu')))
    model.add(TimeDistributed(Conv2D(64, (3,3), activation= 'relu')))
##
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))  
    '''
    model.add(TimeDistributed(Conv2D(16, (3,3), activation= 'relu'),input_shape=(None,Width,Height,1)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(32, (3,3), activation= 'relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))  
    model.add(TimeDistributed(Conv2D(64, (3,3), activation= 'relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))      
    model.add(TimeDistributed(Conv2D(64, (3,3), activation= 'relu')))
    #model.add(TimeDistributed(Conv2D(64, (3,3), activation= 'relu')))
    model.add(TimeDistributed(Conv2D(128, (3,3), activation= 'relu')))
    
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))      
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128))
    model.add(Dropout(0.5)) #added
    
    model.add(Dense(256))
    model.add(Dropout(0.5)) #added
  
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    print(model.summary())
    #lot_model(model, to_file='model.png', show_shapes=True)
   
    checkpoint = ModelCheckpoint(filepath='jupter\\out\\MLP_best5.h5', monitor='val_acc',verbose=1, save_best_only=True) #最好模型保存
    #earlyStopping=EarlyStopping(monitor='val_acc', patience=30, verbose=1, mode='auto')    #提早结束迭代
    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, 
                                                                factor=0.5, min_lr=0.00005)#0.0005     学习率自动调整
    
    
    CallBack = [learning_rate_reduction,checkpoint]#,earlyStopping
    #model = load_model('MLP_best5.h5')     #读取最好模型

       
    cm = 0#1                             #修改这个参数可以多次训练
    cm_str = '' if cm==0 else str(cm)
    cm2_str = '' if (cm+1)==0 else str(cm+1)  
    if cm >= 1:
        model.load_weights(os.path.join(pic_dir_out,'data_channel'+cm_str+'.h5'))#加载保存的权重
        #model.load_weights(os.path.join(pic_dir_out,'cnn_model_Cifar10_'+cm_str+'.h5'))    

   
    history = model.fit(X_train,y_train,batch_size=128,epochs=100,validation_data=(X_test, y_test),
                    shuffle=1,verbose=2,
                    callbacks=CallBack
                    )    
    
    # list all data in history
    print(history.history.keys())
    plt.ylim((0, 1))
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('CNN and LSTM accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    plt.savefig('jupter\\out\\CNN+LSTM accuracy.png', dpi=300)
    plt.show()
    
    plt.ylim((0, 1))
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('CNN and LSTM loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('jupter\\out\\CNN+LSTM loss.png', dpi=300)
    plt.show()
    model.save_weights(os.path.join(pic_dir_out,'data_channel'+cm2_str+'.h5'))  
    plot_model(model, to_file='jupter\\out\\CNN+LSTM.png', show_shapes=True) 
#######################################################    
    #model = load_model('MLP_best5.h5')     #读取最好模型
    
    print('\nTesting ------------')     #对测试集进行评估，额外获得metrics中的信息    
    loss, accuracy = model.evaluate(X_test, y_test)
    print('\n')
    print('test loss: ', loss)
    print('test accuracy: ', accuracy)
    #结果预测
    y_prediction = model.predict(X_test)
    #混淆矩阵
    x = np.argmax(y_test,axis=1)
    y= np.argmax(y_prediction,axis=1)
    cm = confusion_matrix(x, y)
    print(cm)
    print('TPR:    0        4        3     ',cm[0][0]/sum(cm[0]),cm[1][1]/sum(cm[1]),cm[2][2]/sum(cm[2]))
    print('FPR:    0        4        3     ',cm[0][2]/sum(cm[0]),cm[1][0]/sum(cm[1]),cm[2][1]/sum(cm[2]))
    
if __name__ == '__main__':

    main()


    