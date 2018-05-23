# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 21:56:23 2018

@author: zx
"""

import pandas as pd
import os
import pickle
import numpy as np
#pkl_file = open('./data/protein_train.pickle', 'rb')


#data1 = pickle.load(pkl_file)

#==============================================================================
# res = []
# for i in data1:
#     new_data = pd.DataFrame(data1[i]).mean()
#     res.append(new_data)
# data = pd.DataFrame(res) 
# data.columns= ["fusong_vec_{0}".format(i) for i in range(0,26)]
# data['Protein_ID'] = data1
# 
#==============================================================================
molecule = open('./data/molecule.pickle','rb')
data2 = pickle.load(molecule,encoding='iso-8859-1')
res2 = []
count = 0
for i in data2:
    if count > 100:
        break
    new_data = pd.DataFrame(data2[i])
    res2.append(new_data.T)
    count += 1 
data = pd.DataFrame(res2) 
data.columns= ["fusong_vec_moc_{0}".format(i) for i in range(0,185)]
data['Molecule_ID'] = data.index
