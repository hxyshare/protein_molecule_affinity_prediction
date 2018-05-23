#-*- coding :utf-8-*-
import random   
import os.path
import path
import datetime
import pandas as pd
import pickle
from gensim.models import Word2Vec  
import lightgbm as lgb
import re
import gensim 
import numpy as np
data_path = './data'
os.chdir(data_path)#设置当前工作空间
print (os.getcwd())#获得当前工作目录
print("reading data")
data = pd.read_csv("data2.csv")
print("reading done")
print("data shape",data.shape)
data.drop("Unnamed: 0",axis=1,inplace=True)

from sklearn.utils import shuffle  
data = shuffle(data)
#print("null",data[data['Ki']>-11].isnull().any())
train_feat = data[data['Ki']> -11].fillna(0)
train_feat = train_feat[(train_feat.Ki>=0)& (train_feat.Ki<=12)]
print("train_feat",train_feat)
for i in range(5,10):
        import random
        random.seed(2018)
        slice1 = random.sample(list(train_feat[(train_feat.Ki>=i) & (train_feat.Ki<=(i+1))].index),8000)
        print(i,"slice1",len(slice1))
        train_feat.drop(slice1,axis=0,inplace=True)
print("under sample ",train_feat.shape)
res = pd.DataFrame()
for i in range(0,5):
        slice1 = np.random.choice(list(train_feat[(train_feat.Ki>=i) & (train_feat.Ki <=(i+1))].index), 800)
        for i in slice1:
            tmp = pd.DataFrame(data.iloc[i])
            res = pd.concat([res,tmp.T])
train_feat = pd.concat([train_feat,res])
res = pd.DataFrame()
for i in range(9,13):
        slice1 = np.random.choice(list(train_feat[(train_feat.Ki>=i) & (train_feat.Ki<=(i+1))].index), 800)
        for i in slice1:
            tmp = pd.DataFrame(data.iloc[i])
            res = pd.concat([res,tmp.T])
print("res shape", res.shape)
train_feat = pd.concat([train_feat,res])
print("over sample",train_feat.shape)

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
    'max_depth': 20,
    'min_data_in_leaf':10,
    'lambda_l2': 3.3,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'learning_rate': 0.003,
    'seed': 2017
    }

num_round = 2000
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
#submission['Ki'][submission["Ki"]<0] = 0 
#submission['Ki'][submission["Ki"]>12] = 12
submission.to_csv(name, index=False)
 ### 特征选择
print('featrue selecting')
df = pd.DataFrame(train_feat.columns.tolist(), columns=['feature'])
df['importance']=list(gbm.feature_importance())                           # 特征分数
df = df.sort_values(by='importance',ascending=False)                      # 特征排序
df.to_csv("feature_score_20180331.csv",index=None,encoding='gbk')  # 保存分数
