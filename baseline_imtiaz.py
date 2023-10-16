import numpy as np # linear algebra
import math
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from matplotlib import pyplot as plt
#from copy import deepcopy
#import math
from sklearn.cluster import KMeans
#from sklearn.manifold import TSNE
#from sklearn.metrics.pairwise import cosine_similarity
#from numpy import dot
#from numpy.linalg import norm
#from imblearn.under_sampling import RandomUnderSampler
#from scipy.spatial import distance
import timeit
from sklearn.metrics.pairwise import euclidean_distances as ecdist
from sklearn.cluster.k_means_ import _init_centroids
from scipy import sparse


def normalizefea(X):
    """
    L2 normalize
    """
    feanorm = np.maximum(1e-14,np.sum(X**2,axis=1))
    X_out = X/(feanorm[:,None]**0.5)
    return X_out


def kmeans_update(X,tmp):
    """
    
    """
    c1 = X[tmp,:].mean(axis = 0)
    
    return c1
def km_le(X,M,method,sigma):
    
    """
    Discretize the assignments based on center
    
    """
    
    e_dist = ecdist(X,M)          
    l = e_dist.argmin(axis=1)
        
    return l

def km_init(X,K,C_init,seed):
    
    """
    Initial seeds
    """
    
    N,D = X.shape
    if isinstance(C_init,str):

        if C_init == 'kmeans_plus':
            M =_init_centroids(X,K,init='k-means++', random_state=seed)
            l = km_le(X,M,None,None)
            
        elif C_init =='kmeans':
            kmeans = KMeans(n_clusters=K).fit(X)
            l =kmeans.labels_
            M = kmeans.cluster_centers_
    else:
        M = C_init.copy(); 
        l = km_le(X,M,None,None)
        
    del C_init

    return M,l

def normalize(S_in):
    maxcol = np.max(S_in, axis=1)
    S_in = S_in - maxcol[:,np.newaxis]
    S_out = np.exp(S_in)
    S_out = S_out/(np.sum(S_out,axis=1)[:,None])
    
    return S_out

def normalize_2(S_in):
    
    S_out = S_in/(np.sum(S_in,axis=1)[:,None])    
    return S_out

def restore_nonempty_cluster (X,K,oldl,oldC,oldS,ts,seed):
        ts_limit = 2
        C_init = 'kmeans'
        if ts>ts_limit:
            print('not having some labels')
            trivial_status = True
            l =oldl.copy();
            C =oldC.copy();
            S = oldS.copy()

        else:
            print('try with new seeds')
            
            C,l =  km_init(X,K,C_init,seed)
            print("CLUSTER restore:")
            print(C)
            print("l restore:")
            print(l)
            sqdist = ecdist(X,C,squared=True)
            S = normalize_2(np.exp((-sqdist)))
            trivial_status = False
        
        return l,C,S,trivial_status


def bound_energy(S, S_in, a_term, b_term, L, bound_lambda, batch = False):
    
    E = (L*np.log(np.maximum(S,1e-20)) - L*np.log(np.maximum(S_in,1e-20)) + a_term*S + bound_lambda*b_term*S).sum()
               
    return E

def compute_b_j(V_j,u_j,S):
    N,K = S.shape
    V_j = V_j.astype('float')
    R_j = u_j*(1/S.sum(axis=0))
    F_j_a = np.tile((u_j*V_j),[K,1]).T
    F_j_b = np.tile(np.dot(V_j,S),[N,1])
    F_j = F_j_a/np.maximum(F_j_b,1e-10)
    result = R_j - F_j
    # print(result[0:10])
    
    return result
    

def bound_update(a_p, X, l, u_V, V_list, bound_lambda, bound_iteration =200, debug=False):
    
    """
    Here in this code, Q refers to Z in our paper.
    """
    start_time = timeit.default_timer()
    print("Inside Bound Update . . .")
    N,K = a_p.shape;
    oldE = float('inf')
    J = len(u_V)
    
        
# Initialize the S

        
    S = np.exp((-a_p))
    S = normalize_2(S)
    
    # Estimate of L
    V_list_float = np.array(V_list).astype('float')
    SV_term = np.dot(V_list_float,S).min(axis=1)
    
    # Max eigen value estimate
    L = max(max(u_V/(SV_term**2))*N,1.0)
    
    for i in range(bound_iteration):
        
        #printProgressBar(i + 1, bound_iteration,length=12)
        
        S = np.maximum(S,1e-12)
        S_in = S.copy()
        
        # Get a and b 
        terms = -a_p.copy()
        b_j_list = [compute_b_j(V_list[j],u_V[j],S) for j in range(J)]
        
#        if np.isnan(np.max(np.array(b_j_list))):
#            print ('check here')
        
        b_term = bound_lambda*(sum(b_j_list))
        terms -= b_term
        terms /= L
        S_in_2 = normalize(terms)  
        S = S_in * S_in_2
        S = normalize_2(S)
        
        if debug:
            print('b_term = {}'.format(b_term[0:10]))
            print('a_p = {}'.format(a_p[0:10]))
            print('terms = {}'.format(terms[0:10]))
            print('S = {}'.format(S[0:10]))
            
#            if np.isnan(np.max(S)):
#                print ('check here')

        E = bound_energy(S, S_in, a_p, b_term, L, bound_lambda)
#        print('Bound Energy {: .4f} at iteration {} '.format(E,i))
        report_E = E
        
        if (i>1 and (abs(E-oldE)<= 5e-4*abs(oldE))):
            print('Converged')
            break

        else:
            oldE = E.copy(); report_E = E      

                        
    elapsed = timeit.default_timer() - start_time
    print('\n Elapsed Time in bound_update', elapsed)
    l = np.argmax(S,axis=1)
    
#    ind = np.argmax(S,axis=0)
#    C= X[ind,:]
    
    return l,S,report_E

def NormalizedCutEnergy(A, S, clustering):
    if isinstance(A, np.ndarray):
        d = np.sum(A, axis=1)

    elif isinstance(A, sparse.csc_matrix):
        
        d = A.sum(axis=1)

    maxclusterid = np.max(clustering)
    #print "max cluster id is: ", maxclusterid
    nassoc_e = 0;
    num_cluster = 0;
    for k in range(maxclusterid+1):
        S_k = S[:,k]
        #print S_k
        if 0 == np.sum(clustering==k):
             continue # skip empty cluster
        num_cluster = num_cluster + 1
        if isinstance(A, np.ndarray):
            nassoc_e = nassoc_e + np.dot( np.dot(np.transpose(S_k),  A) , S_k) / np.dot(np.transpose(d), S_k)
        elif isinstance(A, sparse.csc_matrix):
            nassoc_e = nassoc_e + np.dot(np.transpose(S_k), A.dot(S_k)) / np.dot(np.transpose(d), S_k)
            nassoc_e = nassoc_e[0,0]
    #print "number of clusters: ", num_cluster
    ncut_e = num_cluster - nassoc_e
    return ncut_e

def get_V_jl(x,l,N,K):

    temp =  np.zeros((N,K))
    index_cluster = l[x]
    temp[(x,index_cluster)]=1
    temp = temp.sum(0)
    return temp

def fairness_term_V_j(u_j,S,V_j):
    V_j = V_j.astype('float')
    S_term = np.dot(V_j,S)
    S_term = u_j*np.log(np.maximum(S_term,1e-20))
    term = u_j*np.log(np.maximum(S.sum(0),1e-20)) - S_term
    
    return term

def get_fair_accuracy(u_V,V_list,l,N,K):
    V_j_list  = np.array([get_V_jl(x,l,N,K) for x in V_list])
    
    balance = np.zeros(K)
    J = len(V_list)
    for k in range(K):
        V_j_list_k = V_j_list[:,k].copy()
        balance_temp = np.tile(V_j_list_k,[J,1])
        balance_temp = balance_temp.T/np.maximum(balance_temp,1e-20)
        mask = np.ones(balance_temp.shape, dtype=bool)
        np.fill_diagonal(mask,0)
        balance[k] = balance_temp[mask].min()
        
#    approx_j_per_K = N/(K*V_j_list.shape[0])
#    error = np.abs(V_j_list - approx_j_per_K)
#    error = error.sum()/N
    
    
    return balance.min(), balance.mean()

def get_fair_accuracy_proportional(u_V,V_list,l,N,K):

    V_j_list  = np.array([get_V_jl(x,l,N,K) for x in V_list])
    clustered_uV = V_j_list/sum(V_j_list)
#    balance = V_j_list/sum(V_j_list)
    fairness_error = np.zeros(K)
    u_V =np.array(u_V)
    
    for k in range(K):
        fairness_error[k] = (-u_V*np.log(np.maximum(clustered_uV[:,k],1e-20))+u_V*np.log(u_V)).sum()
    
    return fairness_error.sum()
def compute_energy_fair_clustering(X, C, l, S, u_V, V_list, bound_lambda, A = None, method_cl='kmeans'):
    """
    compute fair clustering energy
    
    """
    J = len(u_V)
    e_dist = ecdist(X,C,squared =True)
    N,K = S.shape
    if method_cl =='kmeans':
        # K-means energy
        clustering_E = (S*e_dist).sum()
    elif method_cl =='ncut':
        
        clustering_E = NormalizedCutEnergy(A,S,l)
    
    # Fairness term 
    fairness_E = [fairness_term_V_j(u_V[j],S,V_list[j]) for j in range(J)]
    fairness_E = (bound_lambda*sum(fairness_E)).sum()
    
    E = clustering_E + fairness_E
    

    return E, clustering_E, fairness_E


def get_S_discrete(l,N,K):
    x = range(N)
    temp =  np.zeros((N,K),dtype=float)
    index_cluster = l[x]
    temp[(x,index_cluster)]=1
#    temp = temp.sum(0)
    return temp

def KernelBound(A, K, S, current_clustering):
    N = current_clustering.size
    unaries = np.zeros((N, K), dtype=np.float)
    d = A.sum(axis=1)
    for i in range(K):

        S_i = S[:,i]
        volume_s_i = np.dot(np.transpose(d), S_i)
        volume_s_i = volume_s_i[0,0]
        #print volume_s_i
        temp = np.dot(np.transpose(S_i), A.dot(S_i)) / volume_s_i / volume_s_i
        temp = temp * d
        #print temp.shape
        temp2 = temp + np.reshape( - 2 * A.dot(S_i) / volume_s_i, (N,1))
        #print type(temp2)
        unaries[:,i] = temp2.flatten()
        
    return unaries

def fair_clustering(X, K, u_V, V_list, lmbda, seed, fairness = False, method = 'kmeans', C_init = "kmeans_plus", A = None):
    
    """ 
    
    Proposed farness clustering method
    
    """
    N,D = X.shape
    start_time = timeit.default_timer()
    
#    # Generate random centers, here we use sigma and mean to ensure it represent the whole data
#    mean = np.mean(X, axis = 0)
#    #print(type(mean))
#    std = np.std(X, axis = 0)
#    
#    C = np.random.randn(K,D)*std + mean 
#    l = np.zeros(N)
#    distances = np.zeros((N,K))
#    
#    for i in range(K):
#        distances[:,i] = np.linalg.norm(X - C[i], axis=1)    
#
#    l = np.argmin(distances, axis = 1)
    
    
    
    
    C,l =  km_init(X,K,C_init,seed)
    assert len(np.unique(l)) == K
    ts = 0

    trivial_status = False
    S = []
    E_org = []
    E_cluster = []
    E_fair = []
    fairness_error = 0.0
    balance  = 0.0
    oldE = 1e100
    
    for i in range(100):
        oldC = C.copy()
        oldl = l.copy()
        oldS = S.copy()
        #oldFE = fairness_error 
        
        if i == 0:
            
            sqdist = ecdist(X,C,squared=True)
            S = normalize_2(np.exp((-sqdist)))
            a_p = S*sqdist
#                pass
            if method == 'ncut':
                S = get_S_discrete(l,N,K)
                sqdist = KernelBound(A, K, S, l)
                a_p = sqdist.copy()
            
        elif method == 'kmeans':
            
            # TODO: Make it parallel for each k
            
            print ('Inside k-means update')
            
            for k in range(C.shape[0]):
                tmp=np.asarray(np.where(l== k))
                if tmp.size !=1:
                    tmp = tmp.squeeze()
                else:
                    tmp = tmp[0]
                C[[k],] = kmeans_update(X,tmp)
                
            sqdist = ecdist(X,C,squared=True)
            a_p = S*sqdist
            
        
    
            
        if fairness ==True and lmbda!=0.0:
            
            if method == 'kmeans':
                l_check = km_le(X,C,None,None)
           
            
            # Check for empty cluster
            if (len(np.unique(l_check))!=K):
                l,C,S,trivial_status = restore_nonempty_cluster(X,K,oldl,oldC,oldS,ts,seed)
                ts = ts+1
                if trivial_status:
                    break
                
            bound_iterations = 600
            
            
            fairness_error = get_fair_accuracy_proportional(u_V,V_list,l_check,N,K)
            print('fairness_error = {:0.4f}'.format(fairness_error))
                
            l,S,bound_E = bound_update(a_p,X, l, u_V, V_list, lmbda, bound_iterations)
            
            fairness_error = get_fair_accuracy_proportional(u_V,V_list,l,N,K)
            print('fairness_error = {:0.4f}'.format(fairness_error))
        
            
        currentE, clusterE, fairE = compute_energy_fair_clustering(X, C, l, S, u_V, V_list,lmbda, A = A, method_cl=method)    
        E_org.append(currentE)
        E_cluster.append(clusterE)
        E_fair.append(fairE)
        
        
        if (len(np.unique(l))!=K) or math.isnan(fairness_error):
            l,C,S,trivial_status = restore_nonempty_cluster(X,K,oldl,oldC,oldS,ts,seed)
            ts = ts+1
            if trivial_status:
                break

        if (i>1 and (abs(currentE-oldE)<= 1e-4*abs(oldE)) or balance>0.99):
            print('......Job  done......')
            break
            
        
        else:       
            oldE = currentE.copy()
    
    
    elapsed = timeit.default_timer() - start_time
    print(elapsed) 
    E = {'fair_cluster_E':E_org[1:],'fair_E':E_fair[1:],'cluster_E':E_cluster[1:]}
    fairness_error = get_fair_accuracy_proportional(u_V,V_list,l,N,K)
    #print('fairness_error = {:0.4f}'.format(fairness_error))
    print('fairness_error _PRINT= {:0.4f}'.format(fairness_error))
    #print('Fairnes Erro_PRINT:')
    
    return C,l,elapsed,S,E

