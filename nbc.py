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
from sklearn.naive_bayes import MultinomialNB
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
            # if count > 10: break
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
##########################################################################
mnb = MultinomialNB()
mnb.fit(_tfv_train,_class_train)
_pre = mnb.predict(_tfv_test)
target_names = np.unique(_pre)

# report = metrics.classification_report(_class_test, _pre,target_names)
pprint(metrics.classification_report(_class_test, _pre,target_names))
# pprint(report)
##########################################################################
#---------------------------------
#        precision    recall  f1-score   support\n\n
# 0       0.89      0.73      0.80       158\n
# 1       0.84      0.83      0.84       173\n
# 2       0.90      0.82      0.86       216\n
# 3       0.72      0.86      0.78       187\n
# 4       0.92      0.86      0.89       185\n
# 5       0.95      0.83      0.88       208\n
# 6       0.93      0.71      0.80       199\n
# 7       0.93      0.91      0.92       194\n
# 8       0.95      0.97      0.96       189\n
# 9       0.99      0.96      0.97       208\n
# 10       0.96      0.98      0.97       205\n
# 11       0.66      0.99      0.79       169\n
# 12       0.93      0.86      0.89       203\n
# 13       0.96      0.94      0.95       202\n
# 14       0.94      0.93      0.94       209\n
# 15       0.53      0.98      0.69       197\n
# 16       0.78      0.97      0.87       179\n
# 17       0.96      0.97      0.97       194\n
# 18       0.99      0.63      0.77       155\n
# 19       1.00      0.13      0.23       136\n\n
# micro avg       0.86      0.86      0.86      3766\n
# macro avg       0.89      0.84      0.84      3766\n
# weighted avg       0.89      0.86      0.85      3766\n'
#---------------------------------