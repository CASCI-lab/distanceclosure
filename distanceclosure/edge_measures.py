# Distance Backbone Synthesis
# Reroute Index

"""
Edge Measures from Distance Closure
==================
"""

from distanceclosure.backbone import iterative_backbone
import networkx as nx
import numpy as np

def metric_distortion(D, weight='weight'):
    
    G, svals = iterative_backbone(D, weight=weight, kind='metric', distortion=True)
    for u, v in G.edges():
        svals[(u, v)] = 1.0
        
    nx.set_edge_attributes(D, values=svals, name='metric_distortion')
    
    return D


def ultrametric_distortion(D, weight='weight'):
    
    G, svals = iterative_backbone(D, weight=weight, kind='ultrametric', distortion=True)
    for u, v in G.edges():
        svals[(u, v)] = 1.0
        
    nx.set_edge_attributes(D, values=svals, name='ultrametric_distortion')
    
    return D


def dombi_synthesis(D, prox_weight='weight', L=1e-3, R=1e3):
    
    G = D.copy()
    nx.set_edge_attributes(G, name='distance', values={(u, v): (1/w - 1) for u, v, w in G.edges(data=prox_weight)})
    U = iterative_backbone(G, weight='distance', kind='ultrametric', distortion=False)
    
    sorted_edges = merge_sort_dbs(G, L, R, U)
    print(sorted_edges)
    nx.set_edge_attributes(G, name='dombi_synthesis', values={key: val for key, val in sorted_edges})
    
    return G
    
    
def merge_sort_dbs(G, L, R, U):
    
    mid = np.sqrt(L*R)
    nx.set_edge_attributes(G, name='new_distance', values={(u, v): d**mid for u, v, d in G.edges(data='distance')})
    B = iterative_backbone(G, weight='new_distance', kind='metric', distortion=False)
    
    if len(set(B.edges())-set(U.edges())) == 0: # type: ignore
        return [R]
    if len(set(G.edges())-set(B.edges())) == 0: # type: ignore
        return [L]
    if len(set(G.edges())-set(B.edges())-set(U.edges())) == 1: # type: ignore
        return [mid]
    
    
    left_edges = merge_sort_dbs(G, L, mid, U)    
    right_edges = merge_sort_dbs(B, mid, R, U)
    
    return merge(left_edges, right_edges)


def merge(left, right):
    result = []
    i = 0  # Pointer for the left list
    j = 0  # Pointer for the right list

    # Compare elements from both lists and append the smaller one
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # If there are remaining elements in left or right, append them
    result.extend(left[i:])
    result.extend(right[j:])

    return list(result)