# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 21:11:29 2018

@author: zx
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import numpy as np
pkl_file = open('./data/protein_train.pickle', 'rb')
data_path = 'E:/python3/MolecularSelect/data'
os.chdir(data_path)#设置当前工作空间

df_protein_train = pd.read_csv('df_protein_train.csv')#1653
print('loading data...........')
data1 = pickle.load(pkl_file)
res = {}
print("ploting")
for i in data1.values():
    if len(i) not in res :
        res[len(i)] = 1
    else:
        res[len(i)] +=1
plt.figure(1)
plt.subplot(211)
#print(np.array(word_dict)[:,0])
word_dict = sorted(res.items(), key=lambda d: d[0],reverse=False) 
x1 = [j[0] for j in word_dict] 
y1 = [j[1] for j in word_dict] 
plt.axis([0, 1500, 0, 15])
plt.bar(x1,y1,color='rgb')  
plt.show()
