# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 20:01:04 2018

@author: zx
"""

import random
import os.path
import os
import datetime
import numpy as np
import pandas as pd

import lightgbm as lgb

######################################## data read #####################################

#工作空间设置

data_path = './data'
os.chdir(data_path)#设置当前工作空间
print (os.getcwd())#获得当前工作目录


#数据读取
df_protein_train    = pd.read_csv('df_protein_train.csv')#1653
df_protein_test     = pd.read_csv('df_protein_test.csv')#414
protein_concat = pd.concat([df_protein_train,df_protein_test])
#分子数据一共111216个
df_molecule         = pd.read_csv('df_molecule.csv')#111216
df_affinity_train   = pd.read_csv('df_affinity_train.csv')#165084
df_affinity_test    = pd.read_csv('df_affinity_test_toBePredicted.csv')#41383
#分辨测试和训练数据
df_affinity_test['Ki'] = -11

data  =  pd.concat([df_affinity_train,df_affinity_test])

  

###############################################################################################

###########                                 feature                               ############

###############################################################################################

#1、Fingerprint分子指纹处理展开

feat = []

for i in range(0,len(df_molecule)):
    feat.append(df_molecule['Fingerprint'][i].split(','))
    
    
feat = pd.DataFrame(feat)
feat = feat.astype('int')
#改名字
feat.columns=["Fingerprint_{0}".format(i) for i in range(0,167)]
#新建一列，为Molecule_ID
feat["Molecule_ID"] = df_molecule['Molecule_ID']
data = data.merge(feat, on='Molecule_ID', how='left')

#2、df_molecule其他特征处理
#添加其他的特征
feat = df_molecule.drop('Fingerprint',axis=1)
#以data做左连接
data = data.merge(feat, on='Molecule_ID', how='left')

#==============================================================================
# 


#import pandas as pd
#import os
import pickle
#pkl_file = open('protein_train.pickle', 'rb')
# 
#data1 = pickle.load(pkl_file)
#res = []
#for i in data1:
#     new_data = pd.DataFrame(data1[i]).mean()
#     res.append(new_data)
#feat1 = pd.DataFrame(res) 
#feat1.columns= ["fusong_vec_{0}".format(i) for i in range(0,26)]
#feat1['Protein_ID'] = data1
 
#data = data.merge(feat1,on='Protein_ID', how='left')
#molecule = open('molecule.pickle','rb')
#data2 = pickle.load(molecule)
#res2 = []
#for i in data2:
#    new_data = pd.DataFrame(data2[i])
#    res2.append(new_data.T)
#feat1 = pd.DataFrame(res2) 
#feat1.columns= ["fusong_vec_moc_{0}".format(i) for i in range(0,185)]
#feat1['Molecule_ID'] = feat1.index
#
#
#data = data.merge(feat1,on='Molecule_ID', how='left')
#
#print(data)
##==============================================================================
#zhangqifeature
###################################################################

import os
import pandas as pd
train_dir = './train_new/'
test_dir = './test_new/'

#train

fnames = ['{}.feat'.format(i) for i in range(2067)]
count = 1
colum_couont = 0

train_new = pd.DataFrame()

for fname in fnames:
    if os.path.exists((train_dir + str(fname))):
            count += 1
            f = open(train_dir + str(fname))
            res = pd.DataFrame()
            for line in f:
                if colum_couont == 5:
                    colum_couont = 0
                    break
                colum_couont +=1 
                numbers = list(map(float, line.split()))
                a = pd.DataFrame(numbers)
                res = pd.concat([res,a.T],axis=1)
            train_new =pd.concat([train_new,res])
train_new.index =  range(1653)
train_new.columns = ["zhangqi_vec_{0}".format(i) for i in range(0,567)]
train_new['Protein_ID']= df_protein_train['Protein_ID']           
              
data = data.merge(train_new,on='Protein_ID', how='left')         