# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 20:08:22 2016

@author: Kristine
"""

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

df = pd.read_csv("two_group_training.csv")
#df = pd.read_csv("E:/training_data/tut1/tut1/1108train_LOG1.csv")

df=df.as_matrix(columns=None)
df_new=df

n_col = len(df[0])
n_row = len(df)
#
#for i in range(1,n_col):
#    features_temp = df[:,i:i+1]
#    kmeans_temp = KMeans(n_clusters=2,random_state=0).fit(features_temp)
#    k1 = kmeans_temp.labels_.ravel()
#    k1 = pd.DataFrame(k1)
#    k1 = k1.as_matrix(columns=None)
#    df_new[:,i:i+1] = k1
        

#features = df_new[:,1:n_col]
features = df_new[:,2:3]
target = df_new[:,0:1]
target = target.ravel()

real_target1 = target[0:n_row]
#real_target2 = target[77:n_row]

fin = np.zeros(n_row)


kmeans_temp = KMeans(n_clusters=2,random_state=0).fit(features)
test_target = kmeans_temp.labels_.ravel()


count_1 = 0
count_2 = 0

for i in range(0,n_row):
    if (test_target[i] == 0 and real_target1[i] == 0):
        count_1 = count_1+1
    elif (test_target[i] == 0 and real_target1[i] == 1) :
        count_2 = count_2+1
        
#print(count_1)
#print(count_2)
        
if count_1 > count_2:
    print "cluster 1 contain all taxi, %d cars are labeled with 0" % (count_1)
    print"%d sample is clustered to the wrong class" % (count_2)
else:
    print "cluster 1 contain all private cars, %d cars are labeled with 1" % (count_2)  
    print"%d sample is clustered to the wrong class" % (count_1)


        
    
count_1 = 0
count_2 = 0
        
for i in range(0,n_row):   
    if (test_target[i] == 1 and real_target1[i] == 1):
        count_1 = count_1+1
    elif (test_target[i] == 1 and real_target1[i] == 0) :
        count_2 = count_2+1
        
#print(count_1)
#print(count_2)

if count_1 > count_2:
    print "cluster 2 contain all private cars, %d cars are labeled with 1" % (count_1)
    print"%d sample is clustered to the wrong class" % (count_2)
    fin = test_target
else:
    print "cluster 2 contain all taxi, %d cars are labeled with 0" % (count_2)  
    print"%d sample is clustered to the wrong class" % (count_1)
    fin = 1 - test_target


semi_accuracy = np.mean(fin == real_target1)

print "Accuracy is :" 
print(semi_accuracy)

