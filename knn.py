# coding=utf-8
import codecs
import tarfile
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from pprint import pprint
path = 'E:/PythonKnn/20news-18828'
dataset = pd.DataFrame(columns=['name', 'text', 'key'])

cc=0
ss = []
for root, dirs, files in os.walk(path):#取路径下的目录名及文件名
    for dir in dirs:
        if cc>19:break#cc调试中控制加载的数据量，以下将以数字代替class类别名
        test = os.listdir(path + '/' + dir)
        _list = []
        # count = 0
        for tt in test:
            # count = count + 1
            # if count > 200: break
            f = codecs.open(path + '/' + dir + '/' + tt, 'rb')
            r = f.read()
            s = r.decode('utf-8', 'ignore')
            dataset.loc[len(dataset)] = [tt, s, cc]
            ss.append(s)
            _list.append(tt)
        cc=cc+1


#-----------------------
_tfidf = TfidfVectorizer()
_list_train, _list_test = train_test_split(dataset,test_size=0.2)#划分训练集，测试集，0.2为测试集数据占比
_tfv_train = _tfidf.fit_transform(_list_train.iloc[:,1])#对训练集的文本进行tf-idf向量化处理
_tfv_test = _tfidf.transform(_list_test['text']).toarray()#使用训练集的词库，将测试集文本转化为对应的向量表示
#---------------------------------------------------------------

_class_test = []
_class_train= []
target_names =[]
for x in _list_train.iloc[:,2]:
    _class_train.append(x)
for y in _list_test.iloc[:,2]:
    _class_test.append(y)
#提取对应文件集的class为list，方便处理
###############################################################################
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(_tfv_train,_class_train)#对训练集的训练过程
pre = knn.predict(_tfv_test)#预测测试集
#使用metrics.classification_report评价分类结果
target_names = np.unique(pre)
pprint(metrics.classification_report(_class_test, pre,target_names))
###############################################################################
#——-------------------K = 5
##      u'              precision    recall  f1-score   support\n\n
              # 0       0.66      0.90      0.76       154\n
#               1       0.70      0.79      0.74       188\n
#               2       0.68      0.76      0.71       184\n
#               3       0.68      0.67      0.68       237\n
#               4       0.79      0.71      0.75       217\n
#               5       0.77      0.77      0.77       187\n
#               6       0.80      0.58      0.67       220\n
#               7       0.90      0.81      0.86       200\n
#               8       0.90      0.95      0.92       178\n
#               9       0.91      0.88      0.90       223\n
#               10       0.89      0.94      0.91       185\n
#               11       0.86      0.93      0.90       191\n
#               12       0.85      0.74      0.79       189\n
#               13       0.92      0.80      0.86       186\n
#               14       0.93      0.91      0.92       195\n
#               15       0.84      0.84      0.84       197\n
#               16       0.90      0.83      0.87       205\n
#               17       0.78      0.96      0.86       179\n
#               18       0.70      0.80      0.75       136\n
#               19       0.76      0.64      0.70       115\n\n
#        micro avg       0.81      0.81      0.81      3766\n
#        macro avg       0.81      0.81      0.81      3766\n
#     weighted avg       0.81      0.81      0.81      3766\n'
#-----------------------------------------------------------------