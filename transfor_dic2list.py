# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 11:17:46 2018

@author: zx
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import numpy as np
pkl_file = open('E:/python3/MolecularSelect/data/protein_train.pickle', 'rb')
data_path = 'E:/python3/MolecularSelect/data'
os.chdir(data_path)#设置当前工作空间

df_protein_train = pd.read_csv('df_protein_train.csv')#1653
print('loading data...........')
sequence_length =600
df_protein_train['Protein_ID']
print("padding data...........")
res1= []
data1 = pickle.load(pkl_file)
for i in data1.values():
    res = []    
    if (len(i) > sequence_length):
        res = i[:sequence_length]
        res1.append(res)
    else:
        num_padding = sequence_length - len(i)
        res = i + [np.random.rand(26)] * num_padding
        res1.append(res)

        
a = np.array(res1)

a = a[np.newaxis,:,:,:]
a = np.transpose(a,(1,0,2,3))
print(a.shape)
print("write file............. ")
f=open('image_train.pkl','wb')
pickle.dump({'train_data': a},f)
#np.save("a.npy",a)
#b = np.load("a.npy")
f=open('image_train.pkl','rb')  
bb=pickle.load(f)  