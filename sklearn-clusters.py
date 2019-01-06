# coding=utf-8
import codecs
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift,estimate_bandwidth
from sklearn.cluster import SpectralClustering
from sklearn.cluster import hierarchical
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
_data = open('E:\PythonKnn/Tweets.txt','r')
dataset = pd.DataFrame(columns=['text','cluster'])#建立数据框存储数据集
# _lables = set()统计原数据集簇的数量 ----89
for line in _data.readlines():
    line = line.strip()
    x = line.split('"')
    t = x[3]#每一条的text内容
    c = x[6].split()[1][0:-1]#每一条的cluster内容
    # _lables.add(c)
    dataset.loc[len(dataset)] = [t,c]
# print len(_lables)
_tfidfv = TfidfVectorizer()
_tiv_data = _tfidfv.fit_transform(dataset['text'])
#------------KMeans-----------------------------start
# kms =KMeans(n_clusters=85)
# _result_kms = kms.fit(_tiv_data)
# _eval_kms = normalized_mutual_info_score(dataset['cluster'],_result_kms.labels_)
# print _eval_kms#  90-->0.7817098923636209 ;  89-->0.7810733547685412;  88-->0.7927833621798205; 85-->0.7967717926956651;  80-->0.7786994535651568
#------------------------------------------------end


#------------AffinityPropagation----------------start
# aff = AffinityPropagation()
# _result_aff = aff.fit(_tiv_data)
# _eval_aff = normalized_mutual_info_score(dataset['cluster'],_result_aff.labels_)
# print _eval_aff #0.7831387602380028
#------------------------------------------------end


#-----------------MeanShift----------------------start
# bandwidth = 0.9#设置带宽值
# msf = MeanShift(bandwidth)
# msf.fit_predict(_tiv_data.toarray())
# _eval_msf = normalized_mutual_info_score(dataset['cluster'],msf.labels_)
# print _eval_msf   #0.8->0.7153595669914425  #0.9->0.7368766768438576  #1->0.5439011170607866
#--------------------------------------------------end


#--------------------Spectral clustering---------start
# spc = SpectralClustering(n_clusters=91)
# _result_spc = spc.fit(_tiv_data)
# _eval_spc = normalized_mutual_info_score(dataset['cluster'],_result_spc.labels_)
# print _eval_spc#85-->0.6697008691122381;  90-->0.6720738043618484;  91-->0.6893144591052057;0.6780639121450768;0.67715
#--------------------------------------------------end


#-------------------- Ward hierarchical clustering--start
# whc = AgglomerativeClustering(n_clusters=89,linkage="ward")
# _result_whc = whc.fit_predict(_tiv_data.toarray())
# _eval_whc = normalized_mutual_info_score(dataset['cluster'],_result_whc)
# print _eval_whc#0.7800394104591923
#-----------------------------------------------------end


#--------------------------AgglomerativeClustering--start
# agg = AgglomerativeClustering(n_clusters=89,linkage="average")
# _result_agg = agg.fit(_tiv_data.toarray())
# _eval_agg = normalized_mutual_info_score(dataset['cluster'],_result_agg.labels_)
# print _eval_agg#0.8993232774941594
#-----------------------------------------------------end


#----------------------DBSCAN-------------------------start
# _result_DBS = DBSCAN(eps=1.1,min_samples=1).fit(_tiv_data)
# _eval_DBS = normalized_mutual_info_score(dataset['cluster'],_result_DBS.labels_)
# print _eval_DBS #e1.1->0.7736104234688506 m1->0.7542496875513609  m2->0.572203836081289   m3->0.5317410696723648  0.4880804982879795    m5->0.45623167576768364
#-----------------------------------------------------end


#------------------------GaussianMixture-------------start
# mix = GaussianMixture(n_components=89,covariance_type='diag',random_state=42)
# _result_mix = mix.fit_predict(_tiv_data.toarray())
# _eval_mix = normalized_mutual_info_score(dataset['cluster'],_result_mix)
# print _eval_mix #diag 0.7911914044420921
#------------------------------------------------------end