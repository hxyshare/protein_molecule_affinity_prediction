# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:15:21 2018

@author: zx
"""

#coding=utf-8  
import seaborn as sbn  
import pylab as plt  
import theano  
from keras.models import Sequential  
from keras.layers import Dense,Activation  
  
  
from keras.models import Model  
  
model = Sequential()  
model.add(Dense(32, activation='relu', input_dim=100))  
model.add(Dense(16, activation='relu',name="Dense_1"))  
model.add(Dense(1, activation='sigmoid',name="Dense_2"))  
model.compile(optimizer='rmsprop',  
              loss='binary_crossentropy',  
              metrics=['accuracy'])  
  
# Generate dummy data  
import numpy as np  
#假设训练和测试使用同一组数据  
data = np.random.random((1000, 100))  
labels = np.random.randint(2, size=(1000, 1))  
  
# Train the model, iterating on the data in batches of 32 samples  
model.fit(data, labels, epochs=10, batch_size=32)  
#已有的model在load权重过后  
#取某一层的输出为输出新建为model，采用函数模型  
dense1_layer_model = Model(inputs=model.input,  
                                     outputs=model.get_layer('Dense_1').output)  
#以这个model的预测值作为输出  
dense1_output = dense1_layer_model.predict(data)  
  
print (dense1_output.shape ) 
print (dense1_output[0] )