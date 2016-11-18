# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 20:47:40 2016

@author: Kristine
"""

file_object = open(r'new_learning_4.txt','w')
#file_object = open(r'new_learning_2.txt','w')

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
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(features, target)
#clf.score(features[1:10], target[1:10])


#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(features, target)
#clf.feature_importances_

import numpy as np
import pandas as pd

df = pd.read_csv("two_group_training.csv")
#df = pd.read_csv("E:/training_data/tut1/tut1/1108train_LOG1.csv")


df=df.as_matrix(columns=None)
df_new=df


#Cluster features one by one
n_col = len(df[0])
#
#for i in range(1,n_col):
#    features_temp = df[:,i:i+1]
#    kmeans_temp = KMeans(n_clusters=2,random_state=0).fit(features_temp)
#    k1 = kmeans_temp.labels_.ravel()
#    k1 = pd.DataFrame(k1)
#    k1 = k1.as_matrix(columns=None)
#    df_new[:,i:i+1] = k1
#        

features = df_new[:,1:n_col]
target = df_new[:,0:1]
target = target.ravel()

#5 Fold Cross Validation
kf = KFold(n=len(target), n_folds=5, shuffle=True)
cv = 0

# AdaBoost can not work on logisticregression and svm and KNN

#####################   Using  SVM with lienar default method #######################  

seq = ["===========","\t","SVC","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
                             
    model = SVC()                          
    model.fit(tr_features, tr_target)
    #Measuring training and test accuracy
    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)

    print "SVC %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1

#####################   Using  LogisticRegression #######################  

cv = 0
file_object.writelines("\n")
seq = ["===========","\t","LogisticRegression","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
    
    model = LogisticRegression()                            
    model.fit(tr_features, tr_target)  

    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)                         
    print "LogisticRegression %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1

#####################   Using tree.DecisionTreeClassifier ####################### 
   
cv = 0
file_object.writelines("\n")
seq = ["===========","\t","DecisionTreeClassifier","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
    
    model = tree.DecisionTreeClassifier()                             
    model.fit(tr_features, tr_target)  

    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)                         
    print "DecisionTreeClassifier %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1


#####################   Using KNeighborsClassifier with k = 1 ####################### 
   
cv = 0
file_object.writelines("\n")
seq = ["===========","\t","KNeighborsClassifier with k = 1","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
    
    model = KNeighborsClassifier(n_neighbors=1)                            
    model.fit(tr_features, tr_target)  

    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)                         
    print "KNeighborsClassifier %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1   
    
    
#####################   Using KNeighborsClassifier with k = 3 ####################### 
   
cv = 0
file_object.writelines("\n")
seq = ["===========","\t","KNeighborsClassifier with k = 3","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
    
    model = KNeighborsClassifier(n_neighbors=3)                            
    model.fit(tr_features, tr_target)  

    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)                         
    print "KNeighborsClassifier %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1   
    
    
#####################   Using GaussianNB ####################### 
   
cv = 0
file_object.writelines("\n")
seq = ["===========","\t","GaussianNB","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
    
    model = GaussianNB()                            
    model.fit(tr_features, tr_target)  

    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)                         
    print "GaussianNB %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1   
    
    
#####################   Using RandomForestClassifier ####################### 
   
cv = 0
file_object.writelines("\n")
seq = ["===========","\t","RandomForestClassifier","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
    
    model = RandomForestClassifier(random_state=1)                            
    model.fit(tr_features, tr_target)  

    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)                         
    print "RandomForestClassifier %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1   
    
    
#####################   Using GradientBoostingClassifier ####################### 
   
cv = 0
file_object.writelines("\n")
seq = ["===========","\t","GradientBoostingClassifier","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
    
    model = GradientBoostingClassifier()                           
    model.fit(tr_features, tr_target)  

    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)                         
    print "GradientBoostingClassifier %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1   

#############################################################################    
#####################   Using BaggingClassifier #############################
############################################################################# 



###################   BaggingClassifier SVC  #####################################
cv = 0
file_object.writelines("\n")
seq = ["===========","\t","BaggingClassifier SVM","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
    
    model = BaggingClassifier(base_estimator = SVC(), 
                              n_estimators=10, max_samples=1.0, max_features=1.0, 
                              bootstrap=True, bootstrap_features=False, 
                              oob_score=False, warm_start=False, n_jobs=1, 
                              random_state=None, verbose=0)                         
    model.fit(tr_features, tr_target)  

    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)                         
    print "BaggingClassifier SVM %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1   
    

###################   BaggingClassifier LogisticRegression  #####################################
cv = 0
file_object.writelines("\n")
seq = ["===========","\t","BaggingClassifier LogisticRegression","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
    
    model = BaggingClassifier(base_estimator = LogisticRegression(), 
                              n_estimators=10, max_samples=1.0, max_features=1.0, 
                              bootstrap=True, bootstrap_features=False, 
                              oob_score=False, warm_start=False, n_jobs=1, 
                              random_state=None, verbose=0)                         
    model.fit(tr_features, tr_target)  

    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)                         
    print "BaggingClassifier LogisticRegression %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1   
    
    
###################   BaggingClassifier DecisionTreeClassifier  #####################################
cv = 0
file_object.writelines("\n")
seq = ["===========","\t","BaggingClassifier DecisionTreeClassifier","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
    
    model = BaggingClassifier(base_estimator = tree.DecisionTreeClassifier(), 
                              n_estimators=10, max_samples=1.0, max_features=1.0, 
                              bootstrap=True, bootstrap_features=False, 
                              oob_score=False, warm_start=False, n_jobs=1, 
                              random_state=None, verbose=0)                         
    model.fit(tr_features, tr_target)  

    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)                         
    print "BaggingClassifier DecisionTreeClassifier %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1  
 
###################   BaggingClassifier KNeighborsClassifier with k = 1 #####################################
cv = 0
file_object.writelines("\n")
seq = ["===========","\t","BaggingClassifier KNeighborsClassifier with k = 1","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
    
    model = BaggingClassifier(base_estimator = KNeighborsClassifier(n_neighbors=1), 
                              n_estimators=10, max_samples=1.0, max_features=1.0, 
                              bootstrap=True, bootstrap_features=False, 
                              oob_score=False, warm_start=False, n_jobs=1, 
                              random_state=None, verbose=0)                         
    model.fit(tr_features, tr_target)  

    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)                         
    print "BaggingClassifier KNeighborsClassifier  with k = 1 %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1   
    
###################   BaggingClassifier KNeighborsClassifier with k = 3 #####################################
cv = 0
file_object.writelines("\n")
seq = ["===========","\t","BaggingClassifier KNeighborsClassifier with k = 3","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
    
    model = BaggingClassifier(base_estimator = KNeighborsClassifier(n_neighbors=3), 
                              n_estimators=10, max_samples=1.0, max_features=1.0, 
                              bootstrap=True, bootstrap_features=False, 
                              oob_score=False, warm_start=False, n_jobs=1, 
                              random_state=None, verbose=0)                         
    model.fit(tr_features, tr_target)  

    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)                         
    print "BaggingClassifier KNeighborsClassifier  with k = 3 %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1 
    
    
###################   BaggingClassifier GaussianNB #####################################
cv = 0
file_object.writelines("\n")
seq = ["===========","\t","BaggingClassifier GaussianNB","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
    
    model = BaggingClassifier(base_estimator = GaussianNB(), 
                              n_estimators=10, max_samples=1.0, max_features=1.0, 
                              bootstrap=True, bootstrap_features=False, 
                              oob_score=False, warm_start=False, n_jobs=1, 
                              random_state=None, verbose=0)                         
    model.fit(tr_features, tr_target)  

    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)                         
    print "BaggingClassifier GaussianNB  %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1 
    
###################   BaggingClassifier RandomForestClassifier #####################################
cv = 0
file_object.writelines("\n")
seq = ["===========","\t","BaggingClassifier RandomForestClassifier","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
    
    model = BaggingClassifier(base_estimator = RandomForestClassifier(random_state=1), 
                              n_estimators=10, max_samples=1.0, max_features=1.0, 
                              bootstrap=True, bootstrap_features=False, 
                              oob_score=False, warm_start=False, n_jobs=1, 
                              random_state=None, verbose=0)                         
    model.fit(tr_features, tr_target)  

    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)                         
    print "BaggingClassifier RandomForestClassifier  %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1 
    
###################   BaggingClassifier GradientBoostingClassifier #####################################
cv = 0
file_object.writelines("\n")
seq = ["===========","\t","BaggingClassifier GradientBoostingClassifier","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
    
    model = BaggingClassifier(base_estimator = GradientBoostingClassifier(), 
                              n_estimators=10, max_samples=1.0, max_features=1.0, 
                              bootstrap=True, bootstrap_features=False, 
                              oob_score=False, warm_start=False, n_jobs=1, 
                              random_state=None, verbose=0)                         
    model.fit(tr_features, tr_target)  

    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)                         
    print "BaggingClassifier GradientBoostingClassifier  %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1 
    
    
#############################################################################    
#####################   Using AdaBoostClassifier #############################
#############################################################################     
 
###################   AdaBoostClassifier SVC #####################################
cv = 0
file_object.writelines("\n")
seq = ["===========","\t","AdaBoostClassifier SVC","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
    
    model = AdaBoostClassifier(base_estimator=SVC(),
                               n_estimators=10, 
                               learning_rate=0.1, random_state=0,algorithm='SAMME')                       
    model.fit(tr_features, tr_target)  

    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)                         
    print "AdaBoostClassifier SVC  %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1 
    
    
###################   AdaBoostClassifier LogisticRegression #####################################
########################can not work########################################
    
###################   AdaBoostClassifier DecisionTreeClassifier #####################################
cv = 0
file_object.writelines("\n")
seq = ["===========","\t","AdaBoostClassifier DecisionTreeClassifier","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
    
    model = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(),
                               n_estimators=10, 
                               learning_rate=0.1, random_state=0)                       
    model.fit(tr_features, tr_target)  

    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)                         
    print "AdaBoostClassifier DecisionTreeClassifier  %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1 
   
   
###################   AdaBoostClassifier KNeighborsClassifier  #####################################
#########################  can not work #########################################



###################   AdaBoostClassifier GaussianNB  #####################################
cv = 0
file_object.writelines("\n")
seq = ["===========","\t","AdaBoostClassifier GaussianNB","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
    
    model = AdaBoostClassifier(base_estimator=GaussianNB(),
                               n_estimators=10, 
                               learning_rate=0.1, random_state=0)                       
    model.fit(tr_features, tr_target)  

    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)                         
    print "AdaBoostClassifier GaussianNB  %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1 
    
    
###################   AdaBoostClassifier RandomForestClassifier  #####################################
cv = 0
file_object.writelines("\n")
seq = ["===========","\t","AdaBoostClassifier RandomForestClassifier","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
    
    model = AdaBoostClassifier(base_estimator=RandomForestClassifier(),
                               n_estimators=10, 
                               learning_rate=0.1, random_state=0)                       
    model.fit(tr_features, tr_target)  

    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)                         
    print "AdaBoostClassifier RandomForestClassifier  %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1 
    
    
###################   AdaBoostClassifier GradientBoostingClassifier  #####################################
cv = 0
file_object.writelines("\n")
seq = ["===========","\t","AdaBoostClassifier GradientBoostingClassifier","\t","===============","\n"]
file_object.writelines(seq)

for tr, tst in kf:
    #Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]
    
    model = AdaBoostClassifier(base_estimator=GradientBoostingClassifier(),
                               n_estimators=10, 
                               learning_rate=0.1, random_state=0)                       
    model.fit(tr_features, tr_target)  

    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)                         
    print "AdaBoostClassifier GradientBoostingClassifier  %d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    seq = [str(cv+1)," ",
               "Fold Train Accuracy:",str(tr_accuracy),"\t"
               'Test Accuracy:',str(tst_accuracy),"\n"] 
    file_object.writelines(seq)    
    
    cv += 1 
    
    
file_object.closed