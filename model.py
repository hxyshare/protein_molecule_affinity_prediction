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
from sklearn import preprocessing
data_path = './data'
os.chdir(data_path)#设置当前工作空间
print (os.getcwd())#获得当前工作目录
print("reading data")
df_protein_train    = pd.read_csv('df_protein_train.csv')#1653
df_protein_test     = pd.read_csv('df_protein_test.csv')#414
protein_concat = pd.concat([df_protein_train,df_protein_test])
data = pd.read_csv("data.csv")
print("reading done")
print(data.head())
data.drop("Unnamed: 0",axis=1,inplace=True)
need_feature = []
f = open("feature_score_1.csv")
for i in  f.readlines():
    a = i.split(',')
    need_feature.append(a[0])
f.close()

#
#n = 64
#
#texts = [[word for word in re.findall(r'.{3}',document)] 
#                       for document in list(protein_concat['Sequence'])]
#
#print("traing word2vec")
#model = Word2Vec(texts,size=n,window=4,min_count=1,negative=3,
#                         sg=0,sample=0.001,workers=4,iter=15)  
#
#vectors = pd.DataFrame([model[word] for word in (model.wv.vocab)])
#vectors['Word'] = list(model.wv.vocab)
#vectors.columns= ["vec_{0}".format(i) for i in range(0,n)]+["Word"]
#
#wide_vec = pd.DataFrame()
#
#result1=[]
#
#aa = list(protein_concat['Protein_ID'])
#
#for i in range(len(texts)):
#    result2=[]         
#    for w in range(len(texts[i])):
#        result2.append(aa[i])    
#    result1.extend(result2)
#wide_vec['Id'] = result1
#
#result1=[]
#
#for i in range(len(texts)):
#    result2=[]         
#    for w in range(len(texts[i])):
#        result2.append(texts[i][w])    
#    result1.extend(result2)
#wide_vec['Word'] = result1
#del result1,result2
#
#wide_vec = wide_vec.merge(vectors,on='Word', how='left')
#wide_vec = wide_vec.drop('Word',axis=1)
#wide_vec.columns = ['Protein_ID']+["vec_{0}".format(i) for i in range(0,n)]
#
#del vectors
#
#name = ["vec_{0}".format(i) for i in range(0,n)]
#feat = pd.DataFrame(wide_vec.groupby(['Protein_ID'])[name].agg('mean')).reset_index()
#feat.columns=["Protein_ID"]+["mean_ci_{0}".format(i) for i in range(0,n)]
#data = data.merge(feat, on='Protein_ID', how='left')
#


feature_name2index = {x:i for i,x in enumerate(data.columns)}
#drop_col = [feature_name2index[i] for i in need_feature[300:]]
drop_col = [i for i in feature_name2index if i not in need_feature[:500]]
print(data.shape)
data.drop(drop_col,axis=1,inplace=True)
print(data.shape)
#select protein_id for tarin and test
#data augmentation
#print("writing data")
#data.to_csv("data2.csv")

from sklearn.utils import shuffle  
data = shuffle(data)
#print("null",data[data['Ki']>-11].isnull().any())
train_feat = data[data['Ki']> -11].fillna(0)

feature_name2index = {x:i for i,x in enumerate(data.columns)}
#drop_col = [feature_name2index[i] for i in need_feature[300:]]
drop_col = [i for i in feature_name2index if i not in need_feature[:500]]
print(data.shape)
data.drop(drop_col,axis=1,inplace=True)
print(data.shape)

#select protein_id for tarin and test
#data augmentation
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

#train_feat = train_feat[(train_feat.Ki>=0)& (train_feat.Ki<=12)]
print("train_feat",train_feat)
for i in range(5,10):
        import random
        random.seed(2018)
        slice1 = random.sample(list(train_feat[(train_feat.Ki>=i) & (train_feat.Ki<=(i+1))].index),3000)
        print(i,"slice1",len(slice1))
        train_feat.drop(slice1,axis=0,inplace=True)
print("under sample ",train_feat.shape)
#res = pd.DataFrame()
#for i in range(0,5):
#        print(i)
#        slice1 = np.random.choice(list(train_feat[(train_feat.Ki>=i) & (train_feat.Ki <=(i+1))].index), 50)
#        for i in slice1:
#            tmp = pd.DataFrame(data.iloc[i])
#            res = pd.concat([res,tmp.T])
#print("res",res)
#
#train_feat = pd.concat([train_feat,res])
#train_feat = train_feat.reset_index(drop=True)
#res = pd.DataFrame()
#
#for i in range(9,13):
#     print(i)
#     slice1 = np.random.choice(list(train_feat[(train_feat.Ki>=i) & (train_feat.Ki<=(i+1))].index), 50)
#     for i in slice1:
#            tmp = pd.DataFrame(data.iloc[i])
#            res = pd.concat([res,tmp.T])
#print("res shape", res.shape)
##res.drop("Unnamed: 0",axis=1,inplace=True)
#train_feat = pd.concat([train_feat,res])
#train_feat = train_feat.reset_index(drop=True)
#print("over sample",train_feat.shape)
#train_feat = shuffle(train_feat)
#
testt_feat = data[data['Ki']<=-11].fillna(0)
val_feat = train_feat[train_feat["Protein_ID"].isin(range(0,2000,30))]
#val_feat = train_feat[train_feat["Protein_ID"]>1600]
#train_feat = train_feat[train_feat["Protein_ID"]<=1600]
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

#归一化
train_feat = train_feat.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
val_feat = val_feat.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
testt_feat = testt_feat.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
print(train_feat.head())
print(val_feat.head())
print(testt_feat.head())

print("test shape", testt_feat.shape)
train = lgb.Dataset(train_feat, label=label_train)
val  = lgb.Dataset(val_feat, label=label_val,reference=train)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': ['l2_root'],
    'num_leaves': 2 ** 6,
    'max_depth': 30,
    'min_data_in_leaf':10,
    'lambda_l2': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.6 ,
    'learning_rate': 0.01,
    'seed': 2018
    }

num_round =13000
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

gbm.save_model('lightgbm_model'+nowTime)
