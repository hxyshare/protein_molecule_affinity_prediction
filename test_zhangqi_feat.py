# -*- coding: utf-8 -*-

import os
import pandas as pd



data_path = 'E:/python3/MolecularSelect/data'

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


base_dir = './train_new/'

train_dir = os.path.join(base_dir, 'train')
fnames = ['{}.feat'.format(i) for i in range(2067)]
colum_couont = 0

bb = pd.DataFrame()
pid = []
train_seq_filename = []
for fname in fnames:
    #如果存在这个文件的话
    if os.path.exists((base_dir + str(fname))):
            f = open(base_dir + str(fname))
            print(fname[:-5])
            pid.append(int(fname[:-5]))
            train_seq_filename.append(fname[:-5])
            res = pd.DataFrame()
            for line in f:
                #print(len(line.split()))
                numbers = list(map(float, line.split()))
                a = pd.DataFrame(numbers)
                res = pd.concat([res,a.T],axis=1)
                #print(res)
                #print(colum_couont)
            #print(res)
            bb =pd.concat([bb,res])
#==============================================================================
#     if count == 5:
#         break
#==============================================================================
#print(count)       
#print(bb[1:3])
bb.index = range(1653)
bb.columns = ["zhangqi_vec_{0}".format(i) for i in range(0,1790)]

bb['Protein_ID'] = df_protein_train['Protein_ID']
               
print(len(pid))