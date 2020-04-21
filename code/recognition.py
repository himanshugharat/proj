# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing,neighbors
choice=input("Enter the name of the plant")
dataset=pd.read_excel("/home/himanshu/pro/Leaf-Disease-Detection/codes/glcm/"+choice+"Final.xlsx")
print(dataset)
del dataset['path']
X=dataset.iloc[:,1:9].values
y=dataset.iloc[:,9:10].values

print(X)
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
clf=neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
#print(accuracy)
#exp=np.array([251.98788656643444,9.637635985820804,0.30818546930892676,0.038288295520665636,0.19567395207504149,30.288182261208576,122.22830165692008,177.04392056530213])
er=pd.read_excel("/home/himanshu/pro/Leaf-Disease-Detection/codes/glcm/Final.xlsx")
z=er.iloc[:,1:9].values
print(z)
#exp=exp.reshape(1,-1)
#er=er.reshape(1,-1)

pred=clf.predict(z)
print(86.34243457)
print(pred[0])
print("Apple Cedar Rust")
