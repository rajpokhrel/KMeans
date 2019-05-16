import numpy as np
from sklearn.metrics import accuracy_score
from sys import exit
import numpy as np
from random import randint
from copy import deepcopy
from sklearn.metrics import pairwise_distances_argmin
import random
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from numpy.random import RandomState
from sklearn.metrics import accuracy_score

def norm(x):
    """
    >>> Function you should not touch
    """
    max_val = np.max(x, axis=0)
    x = x/max_val
    return x

def rand_center(data,k):
    """
    >>> Function you need to write
    >>> Select "k" random points from "data" as the initial centroids.
    """
    num = data.shape[1]
    centroids = np.zeros((k,num))
    for x in range(num):
        dmin,dmax = np.min(data[:,x]), np.max(data[:,x])
        centroids[:,x] = dmin + (dmax - dmin) * np.random.rand(k)
    return centroids


def converged(centroids1, centroids2):
    set1 = set([tuple(c) for c in centroids1])
    set2 = set([tuple(c) for c in centroids2])
    return (set1 == set2)

def update_centroids(data, centroids, k=3):
    """
    >>> Function you need to write
    >>> check whether centroids1==centroids
    >>> add proper code to handle infinite loop if it never converges
    """
    n = data.shape[0]
    label = np.zeros(n, dtype=np.int32)
    for i in range(n):
            min_dist,min_index = np.inf, -1
            for j in range(k):
                dist = SSE(data[i],centroids[j])
                if dist < min_dist:
                    min_dist, min_index = dist,j
                    label[i] = min_index
    for m in range(k):
        centroids[m] = np.mean(data[label==m],axis=0)
    return centroids,label

def kmeans(data,k=3):
    """
    >>> Function you should not touch
    """
    # step 1:
    centroids = rand_center(data,k)
    converge = False
    converged_counter = 1
    while not converge:
        old_centroids = np.copy(centroids)
        # step 2 & 3
        centroids, label = update_centroids(data, old_centroids, k)
        # step 4
        converge = converged(old_centroids, centroids)
        converged_counter +=1
        if converged_counter == 1000:
            print("infinete loop exiting")
            exit()
    print(">>> final centroids")
    print(centroids)
    return centroids, label

def evaluation(predict, ground_truth):
    """
    >>> use F1 and NMI in scikit-learn for evaluation
    """
    score = f1_score(ground_truth, predict, average='macro')
    res = NMI(ground_truth,predict)
    acc = accuracy_score(ground_truth,predict)
    print("Score: ", score)
    print("NMI: ",res)
    print("Accuracy: ", acc)

def gini(predict, ground_truth):
    """
    >>> use the ground truth to do majority vote to assign a flower type for each cluster
    >>> accordingly calculate the probability of missclassifiction and correct classification
    >>> finally, calculate gini using the calculated probabilities
    """
    vote = np.bincount(predict).argmax()
    unique, counts = np.unique(predict, return_counts=True)
    count_dict = dict(zip(unique, counts))
    prob = count_dict[vote] * 1.0/len(predict)
    wrong_prob = 1 - prob
    gini = 1 - (prob**2) + (wrong_prob**2)
    print ("Gini for Majority Vote Label "+ str(vote) + " is " + str(gini))
    for x in count_dict:
        if x != vote:
            prob = count_dict[vote] * 1.0/len(predict)
            wrong_prob = 1 - prob
            gini = 1 - (prob**2) + (wrong_prob**2)
            print ("Gini of Label " + str(x) + " is " + str(gini))
    
    
def SSE(centroids, data):
    """
    >>> Calculate the sum of squared errors for each cluster
    """
    sse = np.sum((centroids-data)**2)
    return np.sqrt(sse)
