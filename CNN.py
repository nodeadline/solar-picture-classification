# -*- coding: utf-8 -*-
"""
Created on Tue May 21 23:14:30 2019

@author: wangz
"""

import cv2
import numpy as np
from skimage import io
from keras.utils import np_utils#, conv_utils
from keras.backend.common import normalize_data_format
from keras.models import Sequential
from keras.layers import Conv2D,Convolution2D, MaxPooling2D, Flatten, Dropout, Dense, Activation,LSTM,ZeroPadding2D,TimeDistributed,GlobalMaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model
from matplotlib import pyplot
import os
import pickle
import keras
import numpy
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import plot_model
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc,confusion_matrix 


def get_name_list(filepath):                #获取各个类别的名字
    pathDir =  os.listdir(filepath)
    out = []
    for allDir in pathDir:
        if os.path.isdir(os.path.join(filepath,allDir)):
            child = allDir#.decode('gbk')    # .decode('gbk')是解决中文显示乱码问题
            out.append(child)
    return out
    
def eachFile(filepath):                 #将目录内的文件名放入列表中
    pathDir = os.listdir(filepath)
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

########################################################################################################
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
def main():

    global Width, Height, pic_dir_out, pic_dir_data
    Width = 300
    Height =150
    num_classes =3

    #allYears_data          data_out_18

    pic_dir_out = 'jupter\\out\\'# 1994-98_2014_18_out 
    pic_dir_data = 'jupter\\put\\data\\'#1994-98_2014_18_out/  

    #(X_train, y_train), (X_test, y_test) = get_data("data_channel",0.7,data_format='channels_last')

    (X_train, y_train), (X_test, y_test) = get_2data("data_channel",resize=False,data_format='channels_last')

    X_train = X_train/255.              #数据预处理
    X_test = X_test/255.
    print (X_train.shape)
    print(y_train.shape)
    print (X_test.shape)
    print(y_test.shape)
    print(X_train.shape)
    
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)    

###########################################CNN构建    
    model = Sequential()                #CNN构建
    model.add(Convolution2D(
        input_shape=(Width, Height, 1),
        #input_shape=(1, Width, Height),
        filters=32,
        kernel_size=3,
        strides=1,
        padding='valid',     
        data_format='channels_last',
    ))
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size=2,strides=2,data_format='channels_last',))
    
    model.add(Convolution2D(32, 3, strides=1, padding='valid', data_format='channels_last'))
    model.add(Activation('relu'))    
    model.add(MaxPooling2D(2, 2, data_format='channels_last'))
    
    model.add(Convolution2D(64, 3, strides=1, padding='valid', data_format='channels_last'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2, data_format='channels_last'))
    '''
    model.add(Convolution2D(64, 3, strides=1, padding='valid', data_format='channels_last'))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(64, 3, strides=1, padding='valid', data_format='channels_last'))
    model.add(Activation('relu')) 
    '''
    model.add(Convolution2D(128, 3, strides=1, padding='same', data_format='channels_last'))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(2, 2, data_format='channels_last'))
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy']) 
    model.summary()
#############################################       
    print('\nTraining ------------')    #从文件中提取参数，训练后存在新的文件中
    
      
        
    checkpoint = ModelCheckpoint(filepath='jupter\\out\\MLP_best5.h5', monitor='val_acc',verbose=1, save_best_only=True) #最好模型保存
    #earlyStopping=EarlyStopping(monitor='val_acc', patience=30, verbose=1, mode='auto')    #提早结束迭代
    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, 
                                                                factor=0.5, min_lr=0.00005)#0.0005     学习率自动调整
    
    
    CallBack = [learning_rate_reduction,checkpoint]#,earlyStopping
    #model = load_model('MLP_best5.h5')     #读取最好模型
    
    
    #模型训练

    cm =0#1                             #修改这个参数可以多次训练
    cm_str = '' if cm==0 else str(cm)
    cm2_str = '' if (cm+1)==0 else str(cm+1)  
    if cm >= 1:
        model.load_weights(os.path.join(pic_dir_out,'data_channel'+cm_str+'.h5'))#加载保存的权重
        #model.load_weights(os.path.join(pic_dir_out,'cnn_model_Cifar10_'+cm_str+'.h5'))    
    
    history=model.fit(X_train,y_train,batch_size=128,epochs=100,validation_data=(X_test, y_test),
                    shuffle=1,verbose=2,
                    callbacks=CallBack
                    #callbacks=[tfck]
                    )
    
    print(history.history.keys())
    
    plt.ylim((0,1))
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('CNN accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('jupter\\out\\CNN accuracy.png', dpi=300)    
    plt.show()
    
    
    plt.ylim((0,1))
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('CNN loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('jupter\\out\\CNN loss.png', dpi=300)
    plt.show()
    
    model.save_weights(os.path.join(pic_dir_out,'data_channel'+cm2_str+'.h5'))  
    plot_model(model, to_file='jupter\\out\\CNN.png', show_shapes=True) 
#######################################################    
    '''
    #归一化
    test_datagen = ImageDataGenerator(rescale=1./255)    
    
    train_generator = train_datagen.flow_from_directory(
            'E:\\zhiyangwang\\jupter\\put\\data\\train',  # 训练集的文件夹位置
            target_size=(300, 150),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels
    
    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            'E:\\zhiyangwang\\jupter\\put\\data\\test',
            target_size=(300, 150),
            batch_size=batch_size,
            class_mode='binary')
    
    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(train_generator,
                        steps_per_epoch= 1600// batch_size,#X_train.shape[0]
                        epochs=100,
                        validation_data=validation_generator,
                        #validation_steps=1600 // batch_size,
                        verbose=1,
                        max_q_size=100,
                        callbacks=[lr_reducer, early_stopper, csv_logger])    
    
    model.save_weights(os.path.join(pic_dir_out,'data_channel'+cm2_str+'.h5'))  
    '''
#######################################################    
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



