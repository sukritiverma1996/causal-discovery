import numpy as np
import networkx as nx
import pandas as pd
import pickle
import sys
from sklearn.metrics import precision_score, recall_score

truth = pd.read_csv('generated_graph_scorebased_target.csv').values
print(truth)
print(type(truth))
print("Computing <precision, recall> results.")

# GES
print("\nGES")
ges = np.load('est_ges.npy')
print(precision_score(truth.ravel(), ges.ravel()), recall_score(truth.ravel(), ges.ravel()))

# SAM v1
print("\nSAM v1")
with open('est_sammod_endtoend.pkl', 'rb') as fp:
    s1 = nx.to_numpy_array(pickle.load(fp))

s1[s1 > 0.5] = 1
s1[s1 <= 0.5] = 0
print(precision_score(truth.ravel(), s1.ravel()), recall_score(truth.ravel(), s1.ravel()))

# SAM v2
print("\nSAM v2")
with open('est_sam_endtoend.pkl', 'rb') as fp:
    s2 = nx.to_numpy_array(pickle.load(fp))

s2[s2 > 0.5] = 1
s2[s2 <= 0.5] = 0
print(precision_score(truth.ravel(), s2.ravel()), recall_score(truth.ravel(), s2.ravel()))

# NOTEARS LINEAR
print("\nNOTEARS")
nt = np.loadtxt('est_notears_linear.csv', delimiter=',')

nt[nt > 0.5] = 1
nt[nt <= 0.5] = 0
print(precision_score(truth.ravel(), nt.ravel()), recall_score(truth.ravel(), nt.ravel()))

# NOTEARS MLP
print("\nNOTEARS MLP")
ntmlp = np.loadtxt('est_notears_nonlinear.csv', delimiter=',')

ntmlp[ntmlp > 0.5] = 1
ntmlp[ntmlp <= 0.5] = 0
print(precision_score(truth.ravel(), ntmlp.ravel()), recall_score(truth.ravel(), ntmlp.ravel()))

