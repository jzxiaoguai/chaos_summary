# -*- coding:utf-8 -*-
from sys import argv
import pandas as pd
import numpy as np

#origin data separator='\t'  data_example: no1 1 0 1 1 1 2 1 2 1 1
#data need to be loaded into an array (format),each tmp is stored in a list,represent an sample.
file_input=open(argv[1],'r')
x=[]
y=[]
for line in file_input:
    tmp=[]
    item=line.split('\t')
    for i in range(9):
        tmp.append(int(item[i+1]))
    label=int(item[-1])
    x.append(tmp)
    y.append(label)

#离散化处理
n=len(x)
order=[int(x[i][1]) for i in range(n)]
level=pd.cut(order,5,labels=[1,2,3,4,5])
for i in range(n):
    x[i][1]=level[i]

#description about y and other varible you are interested
from collections import Counter
freq=Counter(y)
print "label的分布"
print freq

city_id=[int(x[i][0]) for i in range(n)]
print Counter(city_id)

#divide train set and test set
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=1)

#training nb model  MultinomialNB is optional,it depends on your X
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(train_X,train_y)

#predict and evaluate the result
from sklearn.metrics import confusion_matrix
print "========= confusion_matrix ========="
print confusion_matrix(clf.predict(test_X),test_y)

from sklearn.metrics import classification_report
print "===== precision | recall | f1 ======"
print classification_report(clf.predict(test_X), test_y)

#cross validation and evaluate the model
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, x, y, cv=5)
print "====== cross_validation output ======"
print scores
print np.mean(scores)

file_input.close()
