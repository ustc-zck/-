import numpy as np 
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree

from sklearn.model_selection import train_test_split
import skfuzzy as fuzz

import matplotlib.pyplot as plt 

import pydotplus

from sklearn.externals.six import StringIO

import SimpSOM as sps 

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score


import xlrd

#打开excel文件
book = xlrd.open_workbook('票房数据.xlsx')

sheet = book.sheet_by_index(0)

#读取数据
dataset = []
for i in range(sheet.nrows):
    dataset.append(sheet.row_values(i))

X = np.array(dataset)[:,:-1]
X = X.astype(np.float64)

X = normalize(X, axis= 0, norm='max')

y = np.array(dataset)[:,-1]
y = y.astype(np.float64)
np.savetxt('raw_data', X, delimiter=',')
#Kohonen SOM 网络聚类
net = sps.somNet(20,20,X, PBC=True)
net.project(X, printout=True)
list_of_int = net.cluster(X, type = 'qthresh',savefile=True,)

#print(len(list_of_int))
#print(len(list_of_int[1]))

#求聚类的中心点
centers = []
for i in range(len(list_of_int)):
    points = [X[j] for j in list_of_int[i]]
    center = np.mean(points,axis = 0)
    centers.append(center)

np.savetxt('centers.txt', centers, delimiter = ',')

#print(centers)
#print(len(centers))

#每个属性的平均值，也就是三角隶属函数的中心点
mean_center = np.mean(centers, axis = 0)
np.savetxt('center.txt', mean_center, delimiter = ',')
#print(mean_center),这是聚类的中心点，可以打印出来

X_min = np.min(X, axis = 0)
X_max = np.max(X, axis = 0)

#print(X_min)
#print(X_max)

#X_min, center, X_max分别是三角隶属函数的三个分界

X_traingle_membership = [] #三角隶属函数之后的集合
for i in range(len(X[0])):
    traingle_membership_function = fuzz.membership.trimf(X[:,i], np.array([X_min[i], mean_center[i], X_max[i]]))
    X_traingle_membership.append(traingle_membership_function)

#print(X_traingle_membership),这是用三角隶属函数转化后的X,,模糊化的结果
#print(len(X_traingle_membership))
X_data = np.array(X_traingle_membership).transpose()

np.savetxt('fuzzy_data.txt', X_data, delimiter=',')
X_train, X_test, y_train, y_test = train_test_split(X_data, y)


dot_data = StringIO()
#转化X_traingle_membership维度
clf = DecisionTreeClassifier(criterion='gini',random_state=1,min_samples_leaf=0.2)

clf.fit(X_train, y_train)
tree.export_graphviz(clf, out_file = dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('clf.pdf')
print(clf.score(X_test, y_test))
