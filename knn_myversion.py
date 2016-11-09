# -*- coding: utf-8 -*-
"""
Created on Tue Nov 08 17:14:52 2016

@author: ThinkPad User
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold

testdata=[]
label=[]
myfile=open('1108train_1.txt')
for line in myfile.readlines():
    lineEach=line.strip().split('\t')
    testdata.append([float(lineEach[1]),float(lineEach[4]),float(lineEach[7])])
    #label.append([int(lineEach[12])])
"""
testdata=[]
myfile=open('testdata1.txt')
for line in myfile.readlines():
    lineEach=line.strip().split(' ')
    testdata.append([int(lineEach[0]),int(lineEach[1])])
"""

#testdata=np.array(testdata)

"""
x=[[0,1,2,3],[3,1.3,5,2.3],[2,2.4,1,3.5],[6,2.1,2,4]]
y=[2,0,0,2]
"""
label=[0,0,1,1,0,0,0,0,0,1,1,0,0,1,1,1,1,1,1,0,1,1,0,1,0,1,0,1,0,0,1,1,1,1,1,1,1,1,0,1,0,0,0,1,1,1,1,1,1,1]

testdata=np.array(testdata)
label=np.array(label)

#x_train,x_test,y_train,y_test=cross_validation.train_test_split(testdata,label,test_size=0.4,random_state)
#x_train.shape,y_train.shape

#cv = cross_validation.ShuffleSplit(testdata, n_iter=3, test_size=0.2, random_state=0)

"""
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(testdata, label) 


#neigh_score=cross_validation.cross_val_score(neigh,cv=cv)

print(neigh.predict([[232.0, 382.0, 0.0]]))
"""

#5 Fold Cross Validation
kf = KFold(n=len(label), n_folds=5, shuffle=True)

cv = 0
for tr, tst in kf:

    #Train Test Split
    tr_features = testdata[tr, :]
    tr_target = label[tr]

    tst_features = testdata[tst, :]
    tst_target = label[tst]

    #Training Logistic Regression
    # model = LogisticRegression()
    # model.fit(tr_features, tr_target)

    #Training SVM Model
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(tr_features, tr_target)

    #Measuring training and test accuracy
    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)

    print "%d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    cv += 1