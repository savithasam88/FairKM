# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:54:19 2019

@author: 2626469
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from time import sleep
import numpy as np # linear algebra
#import math
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#from matplotlib import pyplot as plt
from copy import deepcopy
#import math
#from sklearn.cluster import KMeans
#from sklearn.manifold import TSNE
#from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
from imblearn.under_sampling import RandomUnderSampler
from scipy.spatial import distance
from baseline_imtiaz import normalizefea,fair_clustering
import pickle
import time
from scipy.stats import wasserstein_distance
from sklearn.metrics import  silhouette_score
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.

#data pre-processing
def getData(X): 
   
    #d = pd.read_csv(filename, header = None)
    #print(d.values)
    df = X
    label = df.filter(items=df.columns[[14]], axis=1)
    label = pd.DataFrame(label)
    #print('LAbels..')
    #print(label.values)
    #df_labels = df.drop(df.columns[[2]],axis=1)
    
    df = df.drop(df.columns[[14,2,0]],axis=1)
    #print('DROPPING:::')
    #print(df.values)

    obj_df = df.select_dtypes(include=['object']).copy()
    #obj_df.head()
    
    onehot = pd.get_dummies(obj_df)
    #onehot.head()

    num_df = df.select_dtypes(include=['number']).copy()
    #for j in range(num_df.shape[1]):
     #   num_df = num_df.replace({j: {'?': ''}})
    #num_df = num_df.fillna(num_df.mean())
    #print(num_df.isnull())
    #num_df.replace({'weight': {'?': ""}}, regex=False)
    #num_df.head()
    
    final_df = pd.concat([onehot, num_df], axis = 1)
    #final_df = final_df.drop(final_df.columns([[101]]), axis=1)
    #final_df.head()
    
    return final_df, label

#for col in onehot.columns: 
#    print(col) 

#to balance the data
def balance(filename):
    d = pd.read_csv(filename, header = None)
    #print(d.values)
    df = pd.DataFrame(d)
    label = df.filter(items=df.columns[[14]], axis=1)
    label = pd.DataFrame(label)

    #print(label)
    #df_labels = df.drop(df.columns[[2]],axis=1)
    
    #df = df.drop(df.columns[[14,2,0]],axis=1)
    #df.head()
    #data, labels = getData(filename)
    dt = df.values
    lab = label.values
    ros = RandomUnderSampler(random_state=27)
    #rus.fit(np.asarray(questions).reshape(-1,1), schemas)
    X_resampled, y_resampled = ros.fit_sample(dt,lab)
    X = pd.DataFrame(X_resampled)
    Y = pd.DataFrame(y_resampled)
    X[4] = pd.to_numeric(X[4], errors='coerce')
    X[10] = pd.to_numeric(X[10], errors='coerce')
    X[11] = pd.to_numeric(X[11], errors='coerce')
    X[12] = pd.to_numeric(X[12], errors='coerce')
    X.to_csv("undersampled_adult_data.csv", sep=',')
    Y.to_csv("undersampled_adult_label.csv", sep=',')
    return X, Y

#Return all datapoints (from data) in a spacific cluster (cluster_ind) based on the labellings given to the points (in clusters) 
def getElements(cluster_ind, data, clusters):
    indices = [i for i, x in enumerate(clusters) if x == cluster_ind]  
    cluster = np.empty((len(indices), data.shape[1]), dtype=object)
   # cluster = np.zeros((len(indices),data.shape[1]))
    ind = 0
    for i in range(data.shape[0]):
        if (i in indices):
            cluster[ind] = data[i]
            ind += 1
    return cluster

#k-no of clusters, list of merit and sensitive or luck attributes, f- is the file pointer
def k_means_naive(data, k, merit,luck,f):
    #print(data)
    n = data.shape[0]
    
    # Number of features in the data
    c = data.shape[1]
    
    
    c1 = len(merit)
    temp_data = np.zeros((n, c1))
   
    for z1 in range(n):
        a = 0
        for z2 in range(c):
            if(z2 not in merit):
                continue
            else:
                temp_data[z1][a] = data[z1][z2]
                a=a+1

    #print(c)
    # Generate random centers, here we use sigma and mean to ensure it represent the whole data
    mean = np.mean(data, axis = 0)
    #print(type(mean))
    std = np.std(data, axis = 0)
    
    
    centers = np.random.randn(k,c)*std + mean
    #print(centers)
    centers_old = np.zeros(centers.shape) # to store old centers
    centers_new = deepcopy(centers) # Store new centers
    #print('Initial centers:')
    #print(centers_new)
    #data.shape
    clusters = np.zeros(n)
    distances = np.zeros((n,k))

    change = np.linalg.norm(centers_new - centers_old)
    #print(error)
    # When, after an update, the estimate of that center stays the same, exit loop
    epoch = 0
    
    while change != 0 :
    # Measure the distance to every center
        for i in range(k):
            #print(data-centers[i])
            distances[:,i] = np.linalg.norm(data - centers_new[i], axis=1)
            
    # Assign all training data to closest center
        clusters = np.argmin(distances, axis = 1)
        
       # print(clusters)
        centers_old = deepcopy(centers_new)
    # Calculate mean for every cluster and update the center
        for i in range(k):
            #print("data")
            #print(data[clusters == i].shape)
            if (data[clusters == i].shape[0] == 0):
                centers_new[i] = np.zeros(c)
            else :
                centers_new[i] = np.mean(data[clusters == i], axis=0)
          
        change = np.linalg.norm(centers_new - centers_old)

        epoch=epoch+1
    
    print('Number of epochs:'+str(epoch))
    
    center_m = np.zeros((k, c1))
    for i in range(k):
            #print(data-centers[i])
            #center_m[i][] = deepcopy(centers[i])
            d= 0
            for j in range(c):
                if (j in luck):
                    continue
                else:
                    center_m[i][d] = centers_new[i][j]
    
    obj_values = {} 
    for w in range(k):
        indices_w = [m for m, n1 in enumerate(clusters) if n1 == w]
        for obj in indices_w:
            obj_values[obj] = np.linalg.norm(temp_data[obj] - center_m[w])
            #print(data-centers[i])
        #distances_m[:,i] = np.linalg.norm(temp_data - center_m[i], axis=1)
    
    #obj_values = np.min(distances_m, axis = 1)
    objective = sum(obj_values.values())
    print('Objective (naive):')
    print(objective)
    silhouette_avg = silhouette_score(temp_data, clusters)
    print('Silhouette avg for naive k-means:')
    print(silhouette_avg)
    f.write('\n')
    f.write('Objective (naive):')
    f.write(str(objective))
    f.write('\n')
    f.write('Silhouette avg for naive k-means:')
    f.write(str(silhouette_avg))
    f.write('\n')
    return centers_new,clusters   

#merit-list of merit attributes, k - no. of clusters,     
def k_means_merit(data, k,merit,f):
   
    #Number of datapoints
    n = data.shape[0]
    # Number of features in the data
    c = data.shape[1]
    
    #Create data with just the merit attributes
    c1 = len(merit)
    temp_data = np.zeros((n, c1))
   
    for z1 in range(n):
        a = 0
        for z2 in range(c):
            if(z2 not in merit):
                continue
            else:
                temp_data[z1][a] = data[z1][z2]
                a=a+1
    
    #to normalize
    #temp_data=normalizefea(temp_data)
    # Generate random centers, here we use sigma and mean to ensure it represent the whole data
    mean = np.mean(temp_data, axis = 0)
    #print(type(mean))
    std = np.std(temp_data, axis = 0)
    
    #Initial centroids
    centers = np.random.randn(k,c1)*std + mean
    centers_old = np.zeros(centers.shape) # to store old centers
    centers_new = deepcopy(centers) # Store new centers
    
    #Clusters is n d giving cluster assignment of each datapoint
    clusters = np.zeros(n)
    #distances is nxk-giving distance of each point to each of the k centroids
    distances = np.zeros((n,k))
    
    #computes difference between prev centroid and current centroid
    change = np.linalg.norm(centers_new - centers_old)
    
    epoch = 0
    
    while change != 0 :
        #Cluster assignment to points based on their distances to the centroids    
        for i in range(k):
            distances[:,i] = np.linalg.norm(temp_data - centers_new[i], axis=1)   
        
        # Assign all training data to closest center
        clusters = np.argmin(distances, axis = 1)
        
        centers_old = deepcopy(centers_new)
       # Calculate mean for every cluster and update the center
        for i in range(k):
            #print("data")
            #print(data[clusters == i].shape)
            if (temp_data[clusters == i].shape[0] == 0):
                centers_new[i] = np.zeros(c1)
            else :
                centers_new[i] = np.mean(temp_data[clusters == i], axis=0)
             
        change = np.linalg.norm(centers_new - centers_old)
        
        epoch=epoch+1
    print('Number of epochs:'+str(epoch))
    #print('Number of epochs:'+str(epoch))
    #Compute the optimization function value; sum of distances of each [point to the centroid of the cluster it is assigned to
    obj_values = {} 
    for w in range(k):
        indices_w = [m for m, n1 in enumerate(clusters) if n1 == w]
        for obj in indices_w:
            obj_values[obj] = np.linalg.norm(temp_data[obj] - centers_new[w])
            #print(data-centers[i])
        #distances_m[:,i] = np.linalg.norm(temp_data - center_m[i], axis=1)
    
    #obj_values = np.min(distances_m, axis = 1)
    objective = sum(obj_values.values())
    
    
    
    print('Objective (merit):')
    print(objective)
    f.write('\n')
    f.write('Objective (merit):')
    f.write(str(objective))
    silhouette_avg = silhouette_score(temp_data, clusters)
    print('Silhouette avg for merit k-means:')
    print(silhouette_avg)
    f.write('\n')
    f.write('Silhouette avg for merit k-means:')
    f.write(str(silhouette_avg))
    f.write('\n') 
    return centers_new,clusters,objective,silhouette_avg   

#Compute dataset centroid based on sensitive attributes attributes-returns DC is c(total no of features) dimensional, where DC[i]=0 if i is not in luck1, else it is average value of the luck attribute across the n points 
def compute_DC_L(data, luck1):
    #print(data)
    n = data.shape[0]
    # Number of features in the data
    c = data.shape[1]
    print(c)
    luck = np.zeros(c)
    for i in range(0, c):
        if (i in luck1):
            luck[i] = 1
        else:
            luck[i] = 0
    dc = np.zeros(c)
    print(luck)
    for j in range(c):
        num = 0.0
        for i in range(n):
            num = num + luck[j] * data[i][j]
        dc[j] = num / n
        
    return dc

#To get a probability distribution of a sensitive attribute like gender - returns a list of size = no of unique values for the luck attribute - each entry = no. of times the value appears in data/n
def getProbabilityDistribution(index, data, values):
    column_count = dict.fromkeys(values, 0)
    for i in range(data.shape[0]):
        column_count[data[i][index]] += 1
    for key in column_count:
        column_count[key] = column_count[key]/data.shape[0]
    data_vector=list(column_count.values())
    return np.asarray(data_vector)

#Compute KL divergence
def KL(a, b):
    
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

#compute KL divergence of two ditributions p1, p2 = avg of KL(p1,p2) and Kl(p2,p1) 
def getDiversion(probVector1, probVector2):
    
    return 0.5 * (KL(probVector1, probVector2) + KL(probVector2, probVector1))
   
#Get initial centroids for data , k clusters, merit = merit attributes
def get_initial_centroids(data,k, merit):
    n = data.shape[0]
    # Number of features in the data
    c = data.shape[1]
    
    c1 = len(merit)
    #Data with just merit attributes
    temp_data = np.zeros((n, c1))
   
    for z1 in range(n):
        a = 0
        for z2 in range(c):
            if(z2 not in merit):
                continue
            else:
                temp_data[z1][a] = data[z1][z2]
                a=a+1
    # Generate random centers, here we use sigma and mean to ensure it represent the whole data
    mean = np.mean(temp_data, axis = 0)
    #print(type(mean))
    std = np.std(temp_data, axis = 0)
    
    centers = np.random.randn(k,c1)*std + mean
    return centers

#Proposed Fair K-means - k -no of clusters, list of merit attributes, DC - centroid based on sensitive attributes, luck_ind - list of sensitive attributes
#f-file pointer to write results, lamb - lambda: parameter that decides the importance of fairness vs. coherence, origin_v and luck_origin are indices of attributes/sensitive attributes in original data (before one-hot representation)
def k_means_fair_modified_f(data, k, merit, DC, luck_ind,f, lamb,origin_v, luck_origin):
    
    
    #print(data)
    #data = data[0:30,:]
    n = data.shape[0]
    # Number of features in the data
    c = data.shape[1]
    #print(c)
    c1 = len(merit)
    #Create data with just merit attributes
    temp_data = np.zeros((n, c1))
   
    for z1 in range(n):
        a = 0
        for z2 in range(c):
            if(z2 not in merit):
                continue
            else:
                temp_data[z1][a] = data[z1][z2]
                a=a+1

    #IF NORMALIZED
    temp_data=normalizefea(temp_data)
    # Generate random centers, here we use sigma and mean to ensure it represent the whole data
    mean = np.mean(temp_data, axis = 0)
    #print(type(mean))
    std = np.std(temp_data, axis = 0)
    
    centers = np.random.randn(k,c1)*std + mean
    #print(centers)
    #centers_old = np.zeros(centers.shape) # to store old centers
    centers_new = deepcopy(centers) # Store new centers

    #Clusters and clusters_fair_new  is used to assign cluster number based on fair to each data point (n sized)
    clusters = np.zeros(n)
    
    distances = np.zeros((n,k))
    #Compute distance of each of the n points from k centroids
    for i in range(k):
        distances[:,i] = np.linalg.norm(temp_data - centers_new[i], axis=1)
        #print(distances)

   
    #Initial assignment in fair is based on merit attributes alone
    clusters = np.argmin(distances, axis = 1)
    
    # Calculate mean for every cluster and update the center
    for i in range(k):
        if (data[clusters == i].shape[0] == 0):
            centers_new[i] = np.zeros(c1)
        else :
            centers_new[i] = np.mean(temp_data[clusters == i], axis=0)
    
    
    #change = np.linalg.norm(centers_new - centers_old)
    
    # When, after an update, the estimate of that center stays the same, exit loop
    epoch = 0
    
    #no of iterations = 5 
    while epoch<5:
        print('Epoch number:'+str(epoch))
        clusters_old = deepcopy(clusters)
        #Try to find a cluster for each object
        for obj in range(n):
            #Calculate distances of objects to centroids..
            for i in range(k):
                distances[:,i] = np.linalg.norm(temp_data - centers_new[i], axis=1) 
            
                        
            #distances_w: a score computed for each cluster when trying to assign object 'obj' to the corresponding cluster
            distances_w = [0 for i in range(k)]
            
            #below three dictionaries are based on prev-iteration fair clustering - in clusters_fair_new
            #for each cluster there is a dictionary from luck attribute to sum of the values of that luck attribute seen in that cluster
            cluster_luck_agg = {}
            #dict from cluster to objects in the cluster
            cluster_objs = {}
            #dict mapping from cluster to sum of distances of the points in that cluster to the cluster centroid
            sum_merit={}
            #print('\n Pre-computing sums::',file=f1)
            for w in range(k):
                 sum_k=0.0
                 #objects in cluster w
                 indices_w = [m for m, n1 in enumerate(clusters) if n1 == w]
                 cluster_objs[w] = indices_w
                 len_w = len(indices_w)
                 #print('\n No of elements in cluster:'+str(w),file=f1)
                 #print('\n LEN:'+str(len_w),file=f1)
                 if (len_w == 0):
                     cluster_luck_agg[w] = {}
                     sum_merit[w]=0.0
                     continue
                 #For cluster w compute the dict luck_agg from luck attr to sum of the values for this luck attr for the objects in that cluster
                 luck_agg = {}
                 for l in luck_ind:
                     agg_l = 0.0
                     for obj_w in indices_w:
                         agg_l = agg_l+data[obj_w][l]
                     luck_agg[l] = agg_l
                     #print('\n Sum of luck index:'+str(l),file=f1)
                     #print(agg_l,file=f1)
                 cluster_luck_agg[w] = luck_agg
            
                #Sum of distances of objects in a cluster to the cluster centroid
            
                 for obj_w in indices_w:
                    sum_k=sum_k+distances[obj_w][w]
                 sum_merit[w]=sum_k
                 #print('\n Sum of dist to centroid cluster:'+str(w),file=f1)
                 #print(sum_k,file=f1)
            
            #Try assigning obj to cluster w-score has a merit part and a luck part    
            for w in range(k):
                #print('\n Computing score when object is assigned to cluster:'+str(w),file=f1)
                #distance of the obj to cluster w's centroid
                dist_merit_w = distances[obj][w]
                #print('\n Distance of object to cluster '+str(w)+' is '+str(dist_merit_w),file=f1)
                luck_agg = cluster_luck_agg[w]
                #print(type(luck_agg))
                indices_w = cluster_objs[w]
                
                    
                len_w = len(indices_w)
                #print('\n No of elements currently in cluster '+str(w)+' is '+str(len_w),file=f1)
                if(len_w!=0.0):
                    #if obj was in cluster w in clusters_fair_new, then the sum of distances of objects in w to its centrioid remains same
                    if obj in indices_w:
                        #print('\n Obj is already in the cluster',file=f1)
                        dist_merit_w = sum_merit[w]
                    #if obj was not in cluster w in clusters_fair_new, then, obj is added to w and the sum of distances of objects in w is incremented by the distance of obj to w's centroid

                    else:
                        #print('\n Obj is not in the cluster..',file=f1)
                        dist_merit_w = sum_merit[w]+distances[obj][w]
                
                #print('\n Assigning obj to cluster '+str(w)+' makes sum of dist of objects in the cluster '+str(dist_merit_w),file=f1)    
                #print()
                #the above steps compute merit part of the score for  wtgh cluster when assigning obj to wth cluster 
                #agg_w stores the luck part of the score - composed of luck scores for each of the k clusters
                agg_w = 0.0
                #if wth cluster is empty
                if not bool(luck_agg):
                    cluster_val=0.0
                    for lo in luck_origin: #for every luck attribute (original)
                        agg_lo=0.0
                        for l in origin_v[lo]:#for every luck index of type lo
                            agg_l = 0.0
                            agg_l = agg_l+ data[obj][l] #adding wobj to w changes value of the luck aggragate
                            #print('\n After assigning obj to cluster: Sum of attr '+str(l)+ 'in cluster '+str(w)+' is'+str(agg_l),file=f1)
                            agg_l = ((agg_l-DC[l])**2)#dev of luck aggregate from dataset avg value for the luck attr
                            agg_lo = agg_lo+agg_l #aggregate over different values for a luck attribute
                        agg_lo=(1.0/len(origin_v[lo]))*agg_lo #multplying above aggreagate acroos different values with (1/no.of values for a luck attr)
                        #print('\n 1/l*dev for cluster '+str(w)+' is'+str(agg_lo),file=f1)
                        cluster_val = cluster_val+agg_lo
                    #nc = cluster_val*(1.0/n)**2
                    #print('\n Normalized with cluster size'+str(w)+' is'+str(nc),file=f1)
                    agg_w = agg_w+cluster_val*(1.0/n)**2 #luck part of the score for wth cluster weighted with (|c|/n)**2
                        
                #if wth cluster is not empty compute luck part of the score for wth cluster    
                else:  
                    
                    flag=0
                    cluster_val=0.0
                    if obj not in indices_w:
                        flag=1
                        for lo in luck_origin:
                            agg_lo=0.0
                            for l in origin_v[lo]:
                                agg_l=0.0
                                agg_l = agg_l+luck_agg[l]
                                agg_l = agg_l+data[obj][l]
                                #print('\n After assigning obj to cluster: Sum of attr '+str(l)+ 'in cluster '+str(w)+' is'+str(agg_l),file=f1)
                                agg_l = agg_l/(len_w+1)
                                
                                agg_l = ((agg_l-DC[l])**2)
                                
                                agg_lo=agg_lo+agg_l
                            agg_lo=(1.0/len(origin_v[lo]))*agg_lo
                            #print('\n 1/l*dev for cluster '+str(w)+' is'+str(agg_lo),file=f1)
                            cluster_val=cluster_val+agg_lo
                    else:
                         for lo in luck_origin:
                            agg_lo=0.0
                            for l in origin_v[lo]:
                                agg_l=0.0
                                agg_l=agg_l+luck_agg[l]
                                #print('\n After assigning obj to cluster: Sum of attr '+str(l)+ 'in cluster '+str(w)+' is'+str(agg_l),file=f1)
                                agg_l = agg_l/(len_w)
                                agg_l = ((agg_l-DC[l])**2)
                                agg_lo=agg_lo+agg_l
                            agg_lo=(1.0/len(origin_v[lo]))*agg_lo
                            #print('\n 1/l*dev for cluster '+str(w)+' is'+str(agg_lo),file=f1)
                            cluster_val=cluster_val+agg_lo
                           
                    #print('\n The deviations '+str(cluster_val),file=f1)      
                    #luck part of the score for wth cluster weighted with (|c|/n)**2
                    if flag==1:
                        agg_w = agg_w+cluster_val*(((len_w+1.0)/n)**2)
                        #nc = cluster_val*(((len_w+1.0)/n)**2)
                        #print('\n Normalized with cluster size:'+str(w)+' is'+str(nc),file=f1)
                    else:
                        agg_w = agg_w+cluster_val*(((len_w+0.0)/n)**2)
                        #nc = cluster_val*(((len_w+0.0)/n)**2)
                        #print('\n Normalized with cluster size:'+str(w)+' is'+str(nc),file=f1)
                        
                
                #print('\n normalized with cluster size - dev score for the cluster '+str(agg_w),file=f1)
                #Computing score of other clusters when obj is assigned to cluster w 
                dist_merit=0.0
                for o in range(k):
                    
                    #agg_o=0.0
                    cluster_val=0.0
                    
                    if w!=o:
                        #print('\n Computing score for cluster '+str(o)+' when assigning obj to cluster '+str(w),file=f1)
                        #getting the statistics for cluster o precomputed
                        sum_dist = sum_merit[o]    
                        luck_agg1 = cluster_luck_agg[o]
                        #print(type(luck_agg))
                        indices_o = cluster_objs[o]
                        
                        len_o = len(indices_o)
                        #if cluster o is empty, merit part and luck part of the score is 0 for cluster o
                        if not bool(luck_agg1):
                            #print('\n Cluster '+str(o)+' is empty',file=f1)
                            cluster_val = 0.0
                            dist_merit_o=0.0
                            #print('\n Hence, sum of dist of obj to centroid '+str(dist_merit_o),file=f1)
                            #print('\n Hence, sum of dev of obj '+str(cluster_val),file=f1)
                        #if o is not empty
                        else:
                            #print('\n Cluster '+str(o)+' is not empty',file=f1)
                            #print('\n No of elements:'+str(len_o),file=f1)
                            flag=0
                            #if obj is not in o
                            if obj not in indices_o:
                                #print('\n Object is not in the cluster',file=f1)
                                flag=1
                                #meit part of the score is sum precomputed 
                                dist_merit_o = sum_dist
                                #print('\n Hence, sum of dist of obj to centroid '+str(dist_merit_o),file=f1)
                                #compute luck part of the score
                                for lo in luck_origin: 
                                    agg_lo=0.0
                                    for l in origin_v[lo]:
                                        agg_l=0.0
                                        agg_l = agg_l+luck_agg1[l]
                                        #print('\n After assigning obj to cluster:'+str(w)+'- Sum of attr '+str(l)+ 'in cluster '+str(o)+' is'+str(agg_l),file=f1)
                                        agg_l = agg_l/len_o
                                        agg_l = ((agg_l-DC[l])**2)#* ((len_o/n)**2)
                                        agg_lo = agg_lo+agg_l
                                    agg_lo=(1.0/len(origin_v[lo]))*agg_lo
                                    #print('\n 1/l*dev for cluster '+str(o)+' is'+str(agg_lo),file=f1)
                                    cluster_val = cluster_val+agg_lo
                                #print('\n Hence, sum of dev of obj '+str(cluster_val),file=f1)      
                            else:#if obj was in o, remove it from o as it has been adde dto w 
                                #print('\n obj is in cluster '+str(o),file=f1)
                                dist_merit_o=sum_dist-distances[obj][o]
                                #print('\n Hence, sum of dist of obj to centroid after removing obj '+str(dist_merit_o),file=f1)
                                for lo in luck_origin: 
                                    agg_lo=0.0
                                    for l in origin_v[lo]:
                                        agg_l=0.0
                                        agg_l = agg_l+luck_agg1[l]
                                        agg_l = agg_l-data[obj][l]
                                        #print('\n After assigning obj to cluster:'+str(w)+'- Sum of attr '+str(l)+ 'in cluster '+str(o)+' is (after removing obj)'+str(agg_l),file=f1)
                                        #if after removing obj, cluster o becomed empty, luck score also becomes 0
                                        if((len_o-1)==0):
                                            agg_l = 0.0
                                         
                                        else:
                                            agg_l = agg_l/(len_o-1.0)
                                            agg_l =((agg_l-DC[l])**2)#* (((len_o-1)/n)**2)
                                        agg_lo = agg_lo+agg_l
                                    agg_lo=(1.0/len(origin_v[lo]))*agg_lo
                                    #print('\n 1/l*dev for cluster '+str(o)+' is'+str(agg_lo),file=f1)
                                    cluster_val=cluster_val+agg_lo
                                #print('\n Hence, sum of dev of obj '+str(cluster_val),file=f1)  
                            
                        #print('Cluster_val-w:'+str(w)+', o:'+str(o)+' val:'+str(cluster_val))
                        #aggreagte luck score for every other cluster o
                           
                                  
                            if flag==1:
                                #print('\n Cluster_val:'+str(cluster_val),file=f1)
                                #nc = cluster_val*(((len_o-0.0)/n)**2)
                                cluster_val =cluster_val*(((len_o-0.0)/n)**2)
                                
                                #norm=(((len_o-0.0)/n)**2)
                                #print('\n Norm:'+str(norm),file=f1)
                                
                                #print('\n Normalized with cluster size:'+str(o)+' is'+str(cluster_val),file=f1)
                            else:
                                #print('\n Cluster_val:'+str(cluster_val),file=f1)
                                cluster_val = cluster_val* (((len_o-1.0)/n)**2)
                                #nc = cluster_val*(((len_o-1.0)/n)**2)
                                #print('\n Normalized with cluster size:'+str(o)+' is'+str(cluster_val),file=f1)
                           
                        #aggregate merit scores for every other cluser o
                        dist_merit=dist_merit+dist_merit_o   
                        agg_w=agg_w+cluster_val
                
                    
                distances_w[w] = (dist_merit_w+dist_merit)+lamb*agg_w
                
            #assign obj to the cluster with min score
            clusters[obj] = np.argmin(distances_w, axis = 0)
            # Calculate new centers based on new cluster assignment in clusters
            #centers_old = deepcopy(centers_new)
            for i in range(k):
                if (data[clusters == i].shape[0] == 0):
                    centers_new[i] = np.zeros(c1)
                else :
                    centers_new[i] = np.mean(temp_data[clusters == i], axis=0)

        if(np.array_equal(clusters_old,clusters)):
            
            epoch=epoch+1
            break
        epoch=epoch+1
    
    print('Number of epochs:'+str(epoch))
   
    #compute merit based value of objective fn
    obj_values = {} 
    for w in range(k):
        indices_w = [m for m, n1 in enumerate(clusters) if n1 == w]
        for obj in indices_w:
            obj_values[obj] = np.linalg.norm(temp_data[obj] - centers_new[w])
    
    objective = sum(obj_values.values())
    
    print('Objective (fair):')
    print(objective)
    f.write('\n')
    f.write('Objective (fair):')
    f.write(str(objective))
    silhouette_avg = silhouette_score(temp_data, clusters)
    print('Silhouette avg for fair k-means:')
    print(silhouette_avg)
    f.write('\n')
    f.write('Silhouette avg for fair k-means:')
    f.write(str(silhouette_avg))
    f.write('\n')
   # f1.close()  
    #f2.close()
    return centers_new,clusters ,objective,silhouette_avg



#compute merit value of objective fn -merit based distances of points to its clsuter centroid
def computeObjective(data,k,clusters, centers_new, merit,f):
    
    n = data.shape[0]
    # Number of features in the data
    c = data.shape[1]
    c1 = len(merit)
    temp_data = np.zeros((n, c1))
   
    for z1 in range(n):
        a = 0
        for z2 in range(c):
            if(z2 not in merit):
                continue
            else:
                temp_data[z1][a] = data[z1][z2]
                a=a+1
    
    #To normalize
    temp_data=normalizefea(temp_data)
    obj_values = {} 
    for w in range(k):
        indices_w = [m for m, n1 in enumerate(clusters) if n1 == w]
        for obj in indices_w:
            obj_values[obj] = np.linalg.norm(temp_data[obj] - centers_new[w])
            #print(data-centers[i])
        #distances_m[:,i] = np.linalg.norm(temp_data - center_m[i], axis=1)
    
    #obj_values = np.min(distances_m, axis = 1)
    objective = sum(obj_values.values())
    f.write('\n')
    f.write('Objective (base): ')
    f.write(str(objective))
    f.write('\n')
    print('Objective (base):')
    print(objective)
    return objective

#to find unique values of an attribute
def uniqueValues(col):
    uniq = []
    for i in range (col.shape[0]):
        #print(col[i])
        if(col[i] not in uniq):
            uniq.append(col[i])
    return uniq


def evaluate_ed(X,k,clusters, luck_origin,f):
    
    
    dt1 = X.values
    sum_card = 0
    attribute_ed = {}
    attri_ed = {}
    attribute_wd = {}
    attri_wd = {}
    for i in range(k):
        cluster_i = getElements(i, dt1, clusters)
        f.write('\n')
        print('Cluster:'+str(i)+' '+str(len(cluster_i)))
        f.write('Cluster:'+str(i)+' '+str(len(cluster_i)))
        sum_card += len(cluster_i)
        if (len(cluster_i) == 0):
            continue
    #print(type(cluster_i))
        
        for ind in luck_origin:
            col = dt1[:,ind]
            uniq = uniqueValues(col)
            vec1 = getProbabilityDistribution(ind, dt1, uniq)
            vec2 = getProbabilityDistribution(ind, cluster_i, uniq)
            ed = distance.euclidean(vec1, vec2)*len(cluster_i)
            ed1 = distance.euclidean(vec1, vec2)
            wd = wasserstein_distance(vec1,vec2)*len(cluster_i)
            wd1 = wasserstein_distance(vec1,vec2)
            
            #cos_sim = dot(vec1, vec2)/(norm(vec1)*norm(vec2))
            #print('Attribute_cosine similarity:'+str(ind))
            #print(cos_sim)
            
            if(ind not in attribute_ed):
                ed_s = []
                ed_s.append(ed)
                attribute_ed[ind] = ed_s
                
                ed_s1 = []
                ed_s1.append(ed1)
                attri_ed[ind] = ed_s1
                
                wd_s = []
                wd_s.append(wd)
                attribute_wd[ind] = wd_s
                
                wd_s1 = []
                wd_s1.append(wd1)
                attri_wd[ind] = wd_s1
                
            else:
                ed_s = attribute_ed[ind]
                ed_s.append(ed)
                attribute_ed[ind] = ed_s
                
                ed_s1 = attri_ed[ind]
                ed_s1.append(ed1)
                attri_ed[ind] = ed_s1
                
                wd_s = attribute_wd[ind]
                wd_s.append(wd)
                attribute_wd[ind] = wd_s
                
                wd_s1 = attri_wd[ind]
                wd_s1.append(wd1)
                attri_wd[ind] = wd_s1
        #print(sum_card)
    
    for i in attribute_ed:
        ed_s = attribute_ed[i]
        wd_s = attribute_wd[i]
        #print(ed_s)
        agg = sum(ed_s)/dt1.shape[0]
        agg_wd = sum(wd_s)/dt1.shape[0]
        ed_s1 = attri_ed[i]
        maxi = max(ed_s1)
        wd_s1 = attri_wd[i]
        maxi_wd = max(wd_s1)
        sum_wd = sum(wd_s1)
        
        f.write('Luck attribute -'+str(i)+' dev(e) -'+str(agg)+'\n')
        f.write('Luck attribute -'+str(i)+' dev(w) -'+str(agg_wd)+'\n')  
        f.write('Luck attribute -'+str(i)+' maxdev(e) -'+str(maxi)+'\n')
        f.write('Luck attribute -'+str(i)+' maxdev(w) -'+str(maxi_wd)+'\n') 
        f.write('Luck attribute -'+str(i)+' sum(w) -'+str(sum_wd)+'\n') 
           
          
        print('Deviation (euclidean) for luck attribute -'+str(i)+' across clusters:'+' '+str(agg))
        print('Deviation (wasserstein) for luck attribute -'+str(i)+' across clusters:'+' '+str(agg_wd))
        print('Maximum deviation (euclidean) for luck attribute -'+str(i)+ ' across clusters:'+str(maxi))
        print('Maximum deviation (wasserstein) for luck attribute -'+str(i)+ ' across clusters:'+str(maxi_wd))
        print('Sum of wasserstein distance for luck attribute -'+str(i)+ ' across clusters:'+str(sum_wd))

#to compute luck attribute deviations from DC
        #luck_origin are luck attribute indices as in original data
def evaluate_edr(X,k,clusters, luck_origin,f,avg):
    
    print('Inside evaluate_edr::')
    dt1 = X.values
    sum_card = 0
    #each attribute->euclidean dista
    attribute_ed = {}
    attri_ed = {} #max euclidean dist
    #each attribute->wasserstein dista
    attribute_wd = {}
    attri_wd = {} #max wasserstein distance
    
    for i in range(k):
        #get elements in cluster i
        cluster_i = getElements(i, dt1, clusters)
        f.write('\n')
        print('Cluster:'+str(i)+' '+str(len(cluster_i)))
        f.write('Cluster:'+str(i)+' '+str(len(cluster_i)))
        sum_card += len(cluster_i)
        if (len(cluster_i) == 0):
            continue
    #print(type(cluster_i))
        
        for ind in luck_origin:
            print('Luck attribute:'+str(ind))
            #get unique values of the luuck attr
            col = dt1[:,ind]
            uniq = uniqueValues(col)
            #print('Unique values of '+str(ind))
            #get prob distr of the luck attr in data
            vec1 = getProbabilityDistribution(ind, dt1, uniq)
            #print('probability distribution in data:')
            #print(vec1)
            #get prob distr of the luck attr in the cluster
            vec2 = getProbabilityDistribution(ind, cluster_i, uniq)
            #print('probability distribution in cluster:'+str(i))
            #print(vec2)
            #find euclidean distance between above two prob distributions- weighted by size of cluster
            ed = distance.euclidean(vec1, vec2)*len(cluster_i)
            #find euclidean distance between above two prob distributions
            ed1 = distance.euclidean(vec1, vec2)
            #find wasserstein distance between above two prob distributions-weighted by size of cluster
            wd = wasserstein_distance(vec1,vec2)*len(cluster_i)
            #find wasserstein distance between above two prob distributions
            wd1 = wasserstein_distance(vec1,vec2)
            
          
            #for each attribute, fill the dictionary attr->euclidean distance for every cluster
            if(ind not in attribute_ed):
                ed_s = []
                ed_s.append(ed)
                attribute_ed[ind] = ed_s
                
                ed_s1 = []
                ed_s1.append(ed1)
                attri_ed[ind] = ed_s1
                
                wd_s = []
                wd_s.append(wd)
                attribute_wd[ind] = wd_s
                
                wd_s1 = []
                wd_s1.append(wd1)
                attri_wd[ind] = wd_s1
                
            else:
                ed_s = attribute_ed[ind]
                ed_s.append(ed)
                attribute_ed[ind] = ed_s
                
                ed_s1 = attri_ed[ind]
                ed_s1.append(ed1)
                attri_ed[ind] = ed_s1
                
                wd_s = attribute_wd[ind]
                wd_s.append(wd)
                attribute_wd[ind] = wd_s
                
                wd_s1 = attri_wd[ind]
                wd_s1.append(wd1)
                attri_wd[ind] = wd_s1
        #print(sum_card)
    
    for i in attribute_ed:
        
        ed_s = attribute_ed[i]
        wd_s = attribute_wd[i]
        #print('ed_s for luck attribute:'+str(i))
        #print(ed_s)
        #sum across clusters
        #print('Dt1.shape():')
        #print(str(dt1.shape[0]))
        agg = sum(ed_s)/dt1.shape[0]
        agg_wd = sum(wd_s)/dt1.shape[0]
        ed_s1 = attri_ed[i]
        #print('ed_s1 for luck attribute:'+str(i))
        #print(ed_s1)
        maxi = max(ed_s1)
        wd_s1 = attri_wd[i]
        maxi_wd = max(wd_s1)
        sum_wd = sum(wd_s1)
        
        f.write('Luck attribute -'+str(i)+' dev(e) -'+str(agg)+'\n')
        f.write('Luck attribute -'+str(i)+' dev(w) -'+str(agg_wd)+'\n')  
        f.write('Luck attribute -'+str(i)+' maxdev(e) -'+str(maxi)+'\n')
        f.write('Luck attribute -'+str(i)+' maxdev(w) -'+str(maxi_wd)+'\n') 
        f.write('Luck attribute -'+str(i)+' sum(w) -'+str(sum_wd)+'\n') 
        val_i=[] 
        val_i.append(agg)
        val_i.append(agg_wd)
        val_i.append(maxi)
        val_i.append(maxi_wd)
        val_i.append(sum_wd)
        if i not in avg:
            avg[i]=val_i
        else:
            val_i_c=avg[i]
            for j in range(0,5):
                val_i_c[j]=val_i_c[j]+val_i[j]
            avg[i]=val_i_c
          
        print('Deviation (euclidean) for luck attribute -'+str(i)+' across clusters:'+' '+str(agg))
        print('Deviation (wasserstein) for luck attribute -'+str(i)+' across clusters:'+' '+str(agg_wd))
        print('Maximum deviation (euclidean) for luck attribute -'+str(i)+ ' across clusters:'+str(maxi))
        print('Maximum deviation (wasserstein) for luck attribute -'+str(i)+ ' across clusters:'+str(maxi_wd))
        print('Sum of wasserstein distance for luck attribute -'+str(i)+ ' across clusters:'+str(sum_wd))
        
    return avg
#comparing two cluster assignments-merit-base, meri-fair  -take every pair check whether they are given same cluster in merit based clustering and fair based clustering      
def computeConfusionMatrix(clusters, clusters_f,n):
    fraction_disagreed = 0.0
    total_pairs = 0
    ss = 0.0
    sd = 0.0
    ds = 0.0
    dd = 0.0
    for i in range(n-1):
        for j in range(i+1,n):
            if(i!=j):
                total_pairs=total_pairs+1
                if(clusters[i]==clusters[j] and clusters_f[i]==clusters_f[j]):
                    ss=ss+1
                elif(clusters[i]==clusters[j] and clusters_f[i]!=clusters_f[j]):
                    sd = sd+1
                elif(clusters[i]!=clusters[j] and clusters_f[i]==clusters_f[j]):
                    ds = ds+1
                elif(clusters[i]!=clusters[j] and clusters_f[i]!=clusters_f[j]):
                    dd = dd+1
    fraction_disagreed = (sd+ds)/total_pairs
    print('ss:')
    print(ss)
    print('sd:')
    print(sd)
    print('ds:')
    print(ds)
    print('dd:')
    print(dd)
    sd = sd/total_pairs
    ds = ds/total_pairs
    return fraction_disagreed,sd,ds            
                

#baseline method
def baseline(data, k, merit, DC, lmbda, luck_ind,f,seed):
    n = data.shape[0]
    # Number of features in the data
    c = data.shape[1]
    #print(c)
#    l_c = 0
#    luck = np.zeros(c)
#    luck_ind = []
#    for i in range(0, c):
#        if (i not in merit):
#            luck[i] = 1
#            luck_ind.append(i)
#            l_c = l_c+1
#        else:
#            luck[i] = 0
            
    #has luck attributes        
    #Create data with just the merit attributes
    c1 = len(merit)
    temp_data = np.zeros((n, c1))
   
    for z1 in range(n):
        a = 0
        for z2 in range(c):
            if(z2 not in merit):
                continue
            else:
                temp_data[z1][a] = data[z1][z2]
                a=a+1

    X = normalizefea(temp_data)
    #X = temp_data
    #get dataste centroid with just the luck attributes
    DC_l=[]
    for l in luck_ind:
        DC_l.append(DC[l])
    #gives proportion of points with each value - if luck attribute has 2 values - list will have two values  =(no.of points with value1/n),(no.of points with value2/n)
    u_V = [x for x in DC_l]  #proportional
    print('u_V:')
    print(u_V) 
    #demography is of size n-gives demography[i]= j if ith point has jth value for the luck attribute 
    demography = np.zeros(n, dtype=int)
    for z1 in range(n):
        
        for l in luck_ind:
            if(data[z1][l] == 1):
                demography[z1] = luck_ind.index(l)
           
    print('Demography:')
    print(demography)
    #V_list has a list for each value of the luck attribute -each list is of size n with value true if the point has the corresponding value else fasle        
    V_list =  [np.array(demography == j) for j in np.unique(demography)]
    print('V_list:')
    print(V_list)  
    fairness = True
    cluster_option = "kmeans"
    
    C,l,elapsed,S,E = fair_clustering(X, k, u_V, V_list, lmbda, seed, fairness, cluster_option, 'kmeans_plus')

    #C=centers, l-cluster assignment
    silhouette_avg = silhouette_score(X, l)
    print('Silhouette avg for base-:')
    print(silhouette_avg)
    f.write('\n')
    f.write('Silhouette avg for base-:')
    f.write(str(silhouette_avg))
    f.write('\n')
    return C,l,silhouette_avg



start_time = time.time()

#Set no. of clusters
k=15 
#index of merit attributes in original data
merit_origin = [1,3,4,6,10,11,12]
#index of merit attributes in one-hot repr
merit=[]
#Merit attribute indices
for i in range(0, 25):
    merit.append(i)
for i in range(32, 47):
    merit.append(i)
for i in range(101, 105):
    merit.append(i)


#index of luck attributes in one-hot repr 
luck=[]
#All 5 luck attr:
for i in range(25, 32):
    luck.append(i)
for i in range(47, 101):
    luck.append(i)

#index of luck attributes in original repr
luck_origin = [5,7,8,9,13]

#mapping from original to one-hot indices
origin_v={}
gender=[58,59]
origin_v[9]=gender
ms=[]
for i in range(25, 32):
    ms.append(i) 
origin_v[5]=ms
rs=[]
for i in range(47, 53):
    rs.append(i) 
origin_v[7]=rs
race=[]
for i in range(53, 58):
    race.append(i) 
origin_v[8]=race
nc=[]
for i in range(60, 101):
    nc.append(i) 
origin_v[13]=nc    
    
#undersampling
X,Y = balance("adult-training.csv")
data, labels = getData(X)
data.to_csv("processed_adult_data.csv", sep=',')

#attribute to be predicted-make it binary
labels = labels.replace({14: {' <=50K': '0', ' >50K': '1'}})
labels.to_csv("processsed_labels.csv", sep=',')
lab = labels.values

unique_array = np.unique(lab)
num_classes = unique_array.shape[0]

#CALLING K-MEANS NAIVE
dt = data.values
f = open("results_fair_all_10.txt", "w")

#Experiment with Naive k-means  
print('CALLING NAIVE::')
#f.write("CALLING NAIVE::\n")
centers_new_naive, clusters_naive = k_means_naive(dt, k, merit,luck, f)
#Evaluate the clusters on fairness and coherence
evaluate_ed(X,k,clusters_naive, luck_origin,f)

#Experiment with S-blind clustering (take average across 100 runs)
avg={}
#Objective is measure of coherence which is the naive k-means objective -  distance of every data point from its centroid
objective_avg=0.0
sil_avg=0.0 #average of silhouette score, another measure of coherence,  across 100 runs
for i in range(0,100):
    np.random.seed(i) 
    print('CALLING MERIT::')
    f.write("\nCALLING MERIT::\n")
    centers_new_blind, clusters_blind, objective, silhouette_avg  = k_means_merit(dt, k, merit,f)
    objective_avg=objective_avg+objective
    sil_avg=sil_avg+silhouette_avg
    ##f.write('\n')
    #evaluate_ed function measures the fairness in terms of deviation - average-euclidean/wasserstein, maximum euclidean/wasserstein  
    avg=evaluate_edr(X,k,clusters_blind, luck_origin,f,avg)
objective_avg=objective_avg/100.0
print('Objective_avg - s blind clustering:'+str(objective_avg))
sil_avg=sil_avg/100.0
print('Silhouette avg - s blind clustering:'+str(sil_avg))
for i in avg:
    val_i=avg[i]
    for j in range(0,5):
        av=val_i[j]/100.0
        print('S-blind clustering - average deviation (average euclidean/wasserstein, max euclidean/wasserstein) for luck attr '+str(i)+'is '+str(av))

DC = compute_DC_L(dt, luck)

#FAIR k-means clustering - average across 2 runs
        
avg={}
objective_avg=0.0
sil_avg=0.0
count=0
for i in range(2):
    np.random.seed(i*100)
    print('CALLING FAIR K-MEANS CONSIDERING 5 SENSITIVE ATTRIBUTES::')
    f.write('\nCALLING FAIR K-MEANS WITH 5 SENSITIVE ATTRIBUTES::\n')
    centers_new_f, clusters_f, objective, silhouette_avg  = k_means_fair_modified_f(dt, k, merit, DC,luck,f,1000000,origin_v, luck_origin)
    objective_avg=objective_avg+objective
    sil_avg=sil_avg+silhouette_avg
    avg=evaluate_edr(X,k,clusters_f, luck_origin,f,avg)
    
objective_avg=objective_avg/2.0
print('Objective_avg:'+str(objective_avg))
sil_avg=sil_avg/2.0
print('Sil_avg:'+str(sil_avg))
for i in avg:
    val_i=avg[i]
    for j in range(0,5):
        av=val_i[j]/2.0
        print(str(av))

#EXPERIMENT WITH SINGLE SENSITIVE ATTRIBUTE 
#EXPERIMENT CONSIDERING GENDER AS SENSITIVE ATTRIBUTE
print('CALLING FAIR K-MEANS WITH GENDER AS SENSITIVE ATTRIBUTE:')
f.write('\nCALLING FAIR WITH GENDER AS LUCK ATTRIBUTE:\n')
luck_origin =[9]
luck=[]
for i in range(58, 60):
    luck.append(i)

DC = compute_DC_L(dt, luck)
avg={}
objective_avg=0.0
sil_avg=0.0
#AVerage across 10 runs
for i in range(10):
    seed=i*100
    centers_new_fg, clusters_fg,objective,silhouette_avg = k_means_fair_modified_f(dt, k, merit, DC,luck,f,1000000,origin_v, luck_origin)
    sil_avg=sil_avg+silhouette_avg
    objective_avg = objective_avg+objective
    avg = evaluate_edr(X,k,clusters_fg, luck_origin,f,avg)
objective_avg=objective_avg/10.0
print('Objective_avg - Fair-K-Means with gender as sensitive attribute:'+str(objective_avg))
sil_avg=sil_avg/10.0
print('Silhouette score average across 10 runs - Fair-K-Means with gender as sensitive attribute:'+str(sil_avg))
for i in avg:
    val_i=avg[i]
    for j in range(0,5):
        av=val_i[j]/10.0
        print(str(av))    

print('CALLING BASELINE WITH GENDER AS SENSITIVE ATTRIBUTE:')
f.write('\nCALLING BASELINE WITH GENDER AS SENSITIVE ATTRIBUTE:\n')
#Average across 50 runs
for i in range(50):
    seed=i*100
    center_bg,clusters_bg,sil= baseline(dt, k, merit, DC, 15.0, luck,f,seed)
    sil_avg=sil_avg+sil
    avg=evaluate_edr(X,k,clusters_bg, luck_origin,f,avg)
    
    obj = computeObjective(dt,k,clusters_bg, center_bg, merit,f)
    objective_avg = objective_avg+obj
objective_avg=objective_avg/50.0
print('Objective_avg - baseline with gender as sensitive attribute:'+str(objective_avg))
sil_avg=sil_avg/50.0
print('Silhouette avg - baseline with gender:'+str(sil_avg))
for i in avg:
    val_i=avg[i]
    for j in range(0,5):
        av=val_i[j]/50.0
        print(str(av))
f.close()

###EXPERIMENT WITH Marital Status - 
#Set sensitive attributes as follows - 
#luck_origin =[5]
#luck=[]
#for i in range(25, 32):
#    luck.append(i)

###EXPERIMENT WITH Relationship status as the only sensitive attribute 
#Set the sensitive attribute as the index of the attribute Relationship status in original csv which is 7
# luck - indicates the indices representing relationship status once original csv is converted to One-hot representation 
#luck_origin =[7]
#luck=[]
#for i in range(47, 53):
#    luck.append(i)
    
###EXPERIMENT WITH race 
#luck_origin =[8]   
#luck=[]
#for i in range(53, 58):
#    luck.append(i)

###EXPERIMENT WITH native country 
#luck_origin =[13]     
#luck=[]
#for i in range(60, 101):
#    luck.append(i)



















