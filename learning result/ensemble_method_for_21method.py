# -*- coding: utf-8 -*-

"""
Created on Wed Nov 09 20:47:40 2016

@author: Kristine
"""
print(__doc__)


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.cluster import KMeans
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
#from sklearn.feature_selection import validation_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
from itertools import product
from sklearn.feature_selection import RFE
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(features, target)
#clf.score(features[1:10], target[1:10])


#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(features, target)
#clf.feature_importances_

import numpy as np
import pandas as pd

df = pd.read_csv("two_group_training.csv")
#df = pd.read_csv("E:/training_data/tut1/tut1/1108train_LOG1.csv")

#df = df[~pd.isna(df)]

df=df.as_matrix(columns=None)
df_new=df

#Cluster features one by one
n_col = len(df[0])

#for i in range(1,n_col):
#    features_temp = df[:,i:i+1]
#    kmeans_temp = KMeans(n_clusters=2,random_state=0).fit(features_temp)
#    k1 = kmeans_temp.labels_.ravel()
#    k1 = pd.DataFrame(k1)
#    k1 = k1.as_matrix(columns=None)
#    df_new[:,i:i+1] = k1
        

features = df_new[:,1:n_col]
target = df_new[:,0:1]
target = target.ravel()


#try voting method

#when using soft vote, svm must use probability=True

clf1 = SVC(probability=True)

clf2 = LogisticRegression()

clf3 = tree.DecisionTreeClassifier()

clf4 = KNeighborsClassifier(n_neighbors=1)

clf5 = KNeighborsClassifier(n_neighbors=3) 
  
clf6 = GaussianNB() 

clf7 = RandomForestClassifier(random_state=1) 

clf8 = GradientBoostingClassifier()

clf9 = BaggingClassifier(base_estimator = SVC(), 
                              n_estimators=10, max_samples=1.0, max_features=1.0, 
                              bootstrap=True, bootstrap_features=False, 
                              oob_score=False, warm_start=False, n_jobs=1, 
                              random_state=None, verbose=0) 
                              
clf10 = BaggingClassifier(base_estimator = LogisticRegression(), 
                              n_estimators=10, max_samples=1.0, max_features=1.0, 
                              bootstrap=True, bootstrap_features=False, 
                              oob_score=False, warm_start=False, n_jobs=1, 
                              random_state=None, verbose=0)  
                              
clf11 = BaggingClassifier(base_estimator = tree.DecisionTreeClassifier(), 
                              n_estimators=10, max_samples=1.0, max_features=1.0, 
                              bootstrap=True, bootstrap_features=False, 
                              oob_score=False, warm_start=False, n_jobs=1, 
                              random_state=None, verbose=0)       
                              
clf12 = BaggingClassifier(base_estimator = KNeighborsClassifier(n_neighbors=1), 
                              n_estimators=10, max_samples=1.0, max_features=1.0, 
                              bootstrap=True, bootstrap_features=False, 
                              oob_score=False, warm_start=False, n_jobs=1, 
                              random_state=None, verbose=0)  
                              
clf13 = BaggingClassifier(base_estimator = KNeighborsClassifier(n_neighbors=3), 
                              n_estimators=10, max_samples=1.0, max_features=1.0, 
                              bootstrap=True, bootstrap_features=False, 
                              oob_score=False, warm_start=False, n_jobs=1, 
                              random_state=None, verbose=0) 
                              
clf14 = BaggingClassifier(base_estimator = GaussianNB(), 
                              n_estimators=10, max_samples=1.0, max_features=1.0, 
                              bootstrap=True, bootstrap_features=False, 
                              oob_score=False, warm_start=False, n_jobs=1, 
                              random_state=None, verbose=0) 
                              
clf15 = BaggingClassifier(base_estimator = RandomForestClassifier(random_state=1), 
                              n_estimators=10, max_samples=1.0, max_features=1.0, 
                              bootstrap=True, bootstrap_features=False, 
                              oob_score=False, warm_start=False, n_jobs=1, 
                              random_state=None, verbose=0)
                              
clf16 = BaggingClassifier(base_estimator = GradientBoostingClassifier(), 
                              n_estimators=10, max_samples=1.0, max_features=1.0, 
                              bootstrap=True, bootstrap_features=False, 
                              oob_score=False, warm_start=False, n_jobs=1, 
                              random_state=None, verbose=0)  
                              
clf17 = AdaBoostClassifier(base_estimator=SVC(probability=True),
                               n_estimators=10, 
                               learning_rate=0.1, random_state=0)   
                               
clf18 = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(),
                               n_estimators=10, 
                               learning_rate=0.1, random_state=0) 
                               
clf19 = AdaBoostClassifier(base_estimator=GaussianNB(),
                               n_estimators=10, 
                               learning_rate=0.1, random_state=0) 

clf20 = AdaBoostClassifier(base_estimator=RandomForestClassifier(),
                               n_estimators=10, 
                               learning_rate=0.1, random_state=0) 
                               
clf21 = AdaBoostClassifier(base_estimator=GradientBoostingClassifier(),
                               n_estimators=10, 
                               learning_rate=0.1, random_state=0)                               
clf1.fit(features,target)
clf2.fit(features,target)
clf3.fit(features,target)
clf4.fit(features,target)
clf5.fit(features,target)
clf6.fit(features,target)
clf7.fit(features,target)
clf8.fit(features,target)
clf9.fit(features,target)
clf10.fit(features,target)
clf11.fit(features,target)
clf12.fit(features,target)
clf13.fit(features,target)
clf14.fit(features,target)
clf15.fit(features,target)
clf16.fit(features,target)
clf17.fit(features,target)
clf18.fit(features,target)
clf19.fit(features,target)
clf20.fit(features,target)
clf21.fit(features,target)

eclf = VotingClassifier(estimators=[('svc', clf1), ('lr', clf2), ('dt', clf3),
                                    ('knc1',clf4),('knc3',clf5),('gnb',clf6),
                                    ('rfc',clf7),('gbc',clf8),('bagsvc',clf9),
                                    ('baglr',clf10),('bagdt',clf11),('bagknn1',clf12),
                                    ('bagknn3',clf13), ('bagnb',clf14),
                                    ('bagrfc',clf15),('baggbc',clf16),('adbsvc',clf17),
                                    ('adbdct',clf18),('adbgnb',clf19),('adbrfc',clf20),
                                    ('adbrfc',clf21)], 
                                    voting='soft')

#eclf = VotingClassifier(estimators=[('rf', clf2), ('gnb', clf3),
#                                    ('svc',clf4),('dtc',clf5),('knc',clf6)], 
#                                    voting='soft')   
#  
#rfe = RFE(eclf, 2)
#rfe = rfe.fit(features, target)                                  
                                    
eclf.fit(features,target)                            

#eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')


for clf, label in zip([clf1, clf2, clf3, clf4, clf5,
                       clf6, clf7, clf8, clf9, clf10,
                       clf11, clf12, clf13, clf14, 
                       clf15,clf16,clf17,clf18, clf19, clf20,
                       clf21,eclf], 
                      ['SVM', 'Logistic Regression', 
                      'Decision Tree', 'KNN k=1','KNN k=3',
                      'GaussianNB','RandomForest','GradientBoosting','Bagging SVM',
                      'Bagging Logistic','Bagging DecisionTree',
                      'Bagging KNN1','Bagging KNN3','Bagging Gnb',
                      'Bagging rfc','Bagging gbc','Adaboost DCT','Adaboost Gnb',
                      'Adaboost rfc','Adaboost gbc','Ensemble']):
    scores = cross_val_score(clf, features, target, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    
    


