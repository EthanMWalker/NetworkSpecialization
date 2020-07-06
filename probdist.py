import multiprocessing as mp
from networkx.generators.geometric import random_geometric_graph as rgg
from networkx import adjacency_matrix as toMatrix
import sparse_specializer as sp
import numpy as np

def compare(trials,args=[],func=rgg):
    #this is creating a list of the sizes of equitable partitions that appear when the algorithm is run trials number of times.
    #exists because multiprocessing library requires a callable function to parallelize.
    if len(args)==0:
        raise ValueError('must contain required arguments')
    stuff = []
    for i in range(trials):
        stuff=stuff+[len(j) for j in sp.DirectedGraph(toMatrix(func(*args))).coloring().values()]
    return stuff
def probs(trials,engines=4,args=[],func=rgg):
    #This is the parallelized version.
    #trials is the number of graphs to create.
    #engines is the number of threads to use in parallelization
    #args is a list of the arguments for the function that is passed in.  ORDER MATTERS
    pool = mp.Pool(engines)
    stuff = pool.apply(compare,args=[trials,args,func])
    return stuff
def compare_radius(r,n,trials):
    #returns a list of all communities generated in int(trials) trials.  This is specifically for random geometric, and will likely be phased out
    stuff = []
    for i in range(trials):
        stuff = stuff + [len(j) for j in sp.DirectedGraph(toMatrix(rgg(n,r))).coloring().values()]
    return stuff

def prob_distr(r,n,trials,threads):
    #returns an empirical probability distribution of the likelihood of a community appearing
    #in a geometric graph with radius r and number of nodes n in trials number of trials.
    pool=mp.Pool(threads)
    stuff = pool.apply(compare_radius,args=[r,n,trials])
    vals,counts = np.unique(stuff,return_counts=True)
    total = 0
    distr=np.zeros(n)
    for i,j in zip(vals,counts):
        total+=i*j
        distr[i-1]=i*j
    return distr/total

#In Progress
#Scmirnov Distance
def Scmirnov(dist1,dist2):
    #these are both distributions with the same number of nodes
    #computes the Scmirnov distance between the distributions
    assert len(dist1) == len(dist2)
    return np.max(np.abs(np.array(dist1)-np.array(dist2)))

#In Progress
#set up "L2" distance between probability distributions by making community size x and probability y
def L2dist(dist1,dist2):
    assert len(dist1) == len(dist2)
    #first turn each distribution into a collection of x,y points
    dist1 = [(i,dist1[i]) for i in range(len(dist1))]
    dist2 = [(i,dist2[i]) for i in range(len(dist2))]
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)
    #for each point in dist1, make a list of the distances between it and each point in dist2
    distances = np.array([[np.linalg.norm(i-j) for i in dist2] for j in dist1])
    return np.max(np.min(distances[distances > 0]))

#making a kernel density estimator for a distribution.
#an implementation using raw graphs
from sklearn.neighbors import KernelDensity as KDE

def rawKDE(n,r,iterations,engines,kernel='gaussian',bandwidth=.3):
    #returns a KernelDensity object fitted to an average distribution style of equitable partitions.
    pool = mp.Pool(engines)
    stuff = pool.apply(compare_radius,args=[r,n,iterations])
    return KDE(kernel=kernel,bandwidth=bandwidth).fit(np.array(stuff).reshape(-1,1))
def toKDE(data,kernel='gaussian',bandwidth=.3):
    #data is a list of community sizes
    return KDE(kernel=kernel,bandwidth=bandwidth).fit(np.array(data).reshape(-1,1))



from matplotlib import pyplot as plt


#figure out how to plot a distribution using the kernel density estimator.
def plot(obj,n,res=None,label=None):
    #obj should be a KernelDensity object, pretrained on some data.
    #n is the number of nodes in the graph that the estimator was trained on, preferably, however, it is merely the x-axis.
    #res is the number of points to evaluate the pdf at.
    if res is None:
        res=n
    domain = np.linspace(0,n,res,endpoint=True)[:,np.newaxis]
    logs = obj.score_samples(domain)
    if label is None:
        plt.plot(domain,np.exp(logs))
    else:
        plt.plot(domain,np.exp(logs),label=label)
    return
