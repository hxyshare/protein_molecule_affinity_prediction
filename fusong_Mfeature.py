# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import pandas as pd
import os
import pickle
#==============================================================================
# pkl_file = open('protein_train.pickle', 'rb')
# 
# data1 = pickle.load(pkl_file)
# res = []
# for i in data1:
#      new_data = pd.DataFrame(data1[i]).mean()
#      res.append(new_data)
# feat1 = pd.DataFrame(res)
# feat1.columns= ["fusong_vec_{0}".format(i) for i in range(0,26)]
# feat1['Protein_ID'] = data1
# 
# data = data.merge(feat1,on='Protein_ID', how='left')
#==============================================================================
print("sdfsdffd")
molecule = open('./data/molecule.pickle','rb')
data2 = pickle.load(molecule,encoding='iso-8859-1')
feat1 = pd.DataFrame(data2).T
feat1.columns= ["fusong_vec_moc_{0}".format(i) for i in range(0,185)]
print(feat1)
