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
import pickle
import lightgbm as lgb
######################################## data read #####################################

#工作空间设置

data_path = './data'
os.chdir(data_path)#设置当前工作空间
print (os.getcwd())#获得当前工作目录
x_train = np.random.random((1000 * 10, 600, 54))

y_train = np.random.random((1000 * 10, 32))

#数据读取
df_protein_train    = pd.read_csv('df_protein_train.csv')#1653
df_protein_test     = pd.read_csv('df_protein_test.csv')#414
#protein_concat = pd.concat([df_protein_train,df_protein_test])
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
#feat = np.array(feat)
print(len(df_molecule))
count = 0 
for i in range(0,len(df_molecule)):
    count += 1
    #feat = np.append(feat,df_molecule['Fingerprint'][i].split(','))
    feat.append(df_molecule['Fingerprint'][i].split(','))
    #print(df_molecule['Fingerprint'][i].split(','))
feat = pd.DataFrame(feat)
feat = feat.astype('int')
#
print("3132342344")  
feat.columns=["Fingerprint_{0}".format(i) for i in range(0,167)]
#新建一列，为Molecule_ID
feat["Molecule_ID"] = df_molecule['Molecule_ID']
data = data.merge(feat, on='Molecule_ID', how='left')
#2、df_molecule其他特征处理
#添加其他的特征
feat = df_molecule.drop('Fingerprint',axis=1)
#以data做左连接
data = data.merge(feat, on='Molecule_ID', how='left')

data = data.merge(feat, on='Molecule_ID', how='left')
data = data.merge(feat, on='Molecule_ID', how='left')
data = data.merge(feat, on='Molecule_ID', how='left')
data = data.merge(feat, on='Molecule_ID', how='left')


#fusong feature
print("fuosng feature")
molecule = open('molecule.pickle','rb')
data2 = pickle.load(molecule,encoding='iso-8859-1')
feat1 = pd.DataFrame(data2).T
feat1.columns= ["fusong_vec_moc_{0}".format(i) for i in range(0,185)]
feat1['Molecule_ID'] = feat1.index

data = data.merge(feat1,on='Molecule_ID', how='left')
molecule.close()
##==============================================================================
#zhangqifeature
###################################################################
print("zhangqifeature")
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
                if colum_couont == 8:
                        colum_couont = 0
                        break
                colum_couont +=1 
                numbers = list(map(float, line.split()))
                a = pd.DataFrame(numbers)
                res = pd.concat([res,a.T],axis=1)
            train_new =pd.concat([train_new,res])
train_new.index =  range(1653)
train_new.columns = ["zhangqi_vec_{0}".format(i) for i in range(0,1090)]
train_new['Protein_ID']= df_protein_train['Protein_ID']           
              
              
#test
count = 1
colum_couont = 0
test_new = pd.DataFrame()

for fname in fnames:
    if os.path.exists((test_dir + str(fname))):
            count += 1
            f = open(test_dir + str(fname))
            res = pd.DataFrame()
            for line in f:
                if colum_couont == 8:
                    colum_couont = 0
                    break 
                colum_couont += 1 
                numbers = list(map(float, line.split()))
                a = pd.DataFrame(numbers)
                res = pd.concat([res,a.T],axis=1)
            test_new =pd.concat([test_new,res])
            f.close()
test_new.index = range(414)
test_new.columns = ["zhangqi_vec_{0}".format(i) for i in range(0,1090)]
test_new['Protein_ID'] = df_protein_test['Protein_ID']

feat1 = pd.concat([train_new,test_new])
#df = pd.DataFrame.convert_objects(convert_numeric=True)
data = data.merge(feat1,on='Protein_ID', how='left')
import gc
del feat1
gc.collect()


#################################### lgb ############################
#need_feature = []
#f = open("feature_score.csv")
#for i in  f.readlines():
#    a = i.split(',')
#    need_feature.append(a[0])
#f.close()

#feature_name2index = {x:i for i,x in enumerate(data.columns)}
#drop_col = [feature_name2index[i] for i in need_feature[300:]]

#drop_col = [i for i in feature_name2index if i not in need_feature[500:]]
#print(data.shape)
#print(data.columns)
#data.drop(data.columns[drop_col],axis=1,inplace=True)
#print(data.shape)
#select protein_id for tarin and test
#data augmentation


from sklearn.utils import shuffle  
data = shuffle(data)
#print("null",data[data['Ki']>-11].isnull().any())
train_feat = data[data['Ki']> -11].fillna(0)

#for index,group in train_feat.groupby("Protein_ID"):
#     print(index)
#     if len(group) > 2000:
#            import random
#            random.seed(2018)
#            slice1 = random.sample(list(group.index),1000)
#            train_feat.drop(slice1,axis=0,inplace=True)
#print("under sample ",train_feat.shape)
#res = pd.DataFrame()
#for index,group in train_feat.groupby("Protein_ID"):
#    print(index)
#    if len(group) < 50 :
#        slice1 = np.random.choice(list(group.index), 10)
#        for i in slice1:
#            tmp = pd.DataFrame(data.iloc[i])
#            res = pd.concat([res,tmp.T])
#print("res shape", res.shape)
#train_feat = pd.concat([train_feat,res])
#print("over sample",train_feat.shape)
#
testt_feat = data[data['Ki']<=-11].fillna(0)
val_feat = train_feat[train_feat["Protein_ID"].isin(range(0,2000,30))]
#train_feat = train_feat[train_feat["Protein_ID"]<=1800]
print("train",len(train_feat),"val",len(val_feat))
print("train shape",train_feat.shape)
label_train  = train_feat['Ki']
label_val  = val_feat['Ki'] 

submission = testt_feat[['Protein_ID','Molecule_ID']]
len(testt_feat)
train_feat = train_feat.drop('Ki',axis=1)
testt_feat = testt_feat.drop('Ki',axis=1)
val_feat = val_feat.drop('Ki',axis=1)
train_feat = train_feat.drop('Protein_ID',axis=1)
testt_feat = testt_feat.drop('Protein_ID',axis=1)
val_feat = val_feat.drop('Protein_ID',axis=1)
train_feat = train_feat.drop('Molecule_ID',axis=1)
testt_feat = testt_feat.drop('Molecule_ID',axis=1)
val_feat = val_feat.drop('Molecule_ID',axis=1)

print("test shape", testt_feat.shape)
train = lgb.Dataset(train_feat, label=label_train)
val  = lgb.Dataset(val_feat, label=label_val,reference=train)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': ['l2_root'],
    'num_leaves': 2 ** 6,
    #'max_depth': 20,
    #'min_data_in_leaf':10,
    'lambda_l2': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'learning_rate': 0.001,
    'seed': 2017
    }

num_round = 100011
gbm = lgb.train(params, 
                  train, 
                  num_round, 
                  verbose_eval=50,
                  valid_sets= [train,val],
                  early_stopping_rounds=150 
                  )

#gbm = lgb.cv(params, train, num_round, nfold=10)

preds_sub = gbm.predict(testt_feat)

#结果保存
print('saving result')
nowTime=datetime.datetime.now().strftime('%m%d%H%M')#现在
name='zx_'+nowTime+'.csv'
submission['Ki'] = preds_sub
submission.to_csv(name, index=False)
 ### 特征选择
print('featrue selecting')
df = pd.DataFrame(train_feat.columns.tolist(), columns=['feature'])
df['importance']=list(gbm.feature_importance())                           # 特征分数
df = df.sort_values(by='importance',ascending=False)                      # 特征排序
df.to_csv("feature_score_20180331.csv",index=None,encoding='gbk')  # 保存分数
