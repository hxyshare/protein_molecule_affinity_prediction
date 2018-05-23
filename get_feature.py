

need_feature = []
from sklearn.utils import shuffle
for i in  open("./data/feature_score_20180331.csv").readlines():
    a = i.split(',')
    need_feature.append(a[0])
print(need_feature)

