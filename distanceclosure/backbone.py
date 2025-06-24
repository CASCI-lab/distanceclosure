# -*- coding: utf-8 -*-
"""
Backbone Subgraph - Fast Implementation
==================

Computes the shortest path distance backbone on a weighted graph.
These algorithms work with edges weighted as distances.
"""

import numpy as np
import networkx as nx
from distanceclosure.dijkstra import single_source_dijkstra_path_length, single_source_target_dijkstra_path
from networkx.algorithms.shortest_paths.weighted import _weight_function

__name__ = 'distanceclosure'
__author__ = """\n""".join(['Rion Brattig Correia <rionbr@gmail.com>', 'Felipe Xavier Costa <fcosta@binghamton.com>'])

__all__ = [
    "metric_backbone",
    "ultrametric_backbone",
    "iterative_backbone",
    "flagged_backbone"
]

__kinds__ = ['metric', 'ultrametric']
__algorithms__ = ['dense', 'dijkstra']

def metric_backbone(D, weight='weight', distortion=False, self_loops=False, cutoff=None, verbose=False, *args, **kwargs):
    """ Alias for :func:`iterative_backbone` with kind=metric.
    """
    
    return iterative_backbone(D, weight=weight, kind='metric', distortion=distortion, self_loops=self_loops, cutoff=cutoff, verbose=verbose, *args, **kwargs)

def ultrametric_backbone(D, weight='weight', distortion=False, self_loops=False, cutoff=None, verbose=False, *args, **kwargs):
    """ Alias for :func:`iterative_backbone`  with kind=ultrametric.
    """
    
    return iterative_backbone(D, weight=weight, kind='ultrametric', distortion=distortion, self_loops=self_loops, cutoff=cutoff, verbose=verbose, *args, **kwargs)

def iterative_backbone(D, weight='weight', kind='metric', distortion=False, self_loops=False, cutoff=None, verbose=False, *args, **kwargs):
    """
    Iterative backbone computation considering node ordering.

    Parameters
    ----------
    D : NetworkX graph
        The Distance graph
    weight : str, optional
        Edge property containing distance values, by default 'weight'
    kind : str, optional
        Distance accumulation kind. Either metric (sum) or ultrametric (max), by default 'metric'
    self_loops : bool, optional
        If the distance graph has nodes with self distance greater than zero, by default False
    cutoff : _type_, optional
        Maximum number of connections in the path. If None, compute the entire closure as is the cutoff is the number of nodes, by default None
    verbose : bool, optional
        Prints statements as it computes, by default False

    Returns
    -------
    NetworkX graph
        The backbone subgraph.

    Raises
    ------
    NotImplementedError
        Self-loop closure and finite step (cutoff) not implemented yet
    """
    
    _check_for_kind(kind)
    
    if self_loops:
        raise NotImplementedError
    if cutoff is not None:
        raise NotImplementedError
    
    if kind == 'metric':
        disjunction = sum
    elif kind == 'ultrametric':
        disjunction = max
    
    G = D.copy()
    weight_function = _weight_function(G, weight)
    
    if verbose:
        total = G.number_of_nodes()
        i = 0
    
    for u in G.nodes():
        if verbose:
            i += 1
            per = i/total
            print("Backbone: Dijkstra: {i:d} of {total:d} ({per:.2%})".format(i=i, total=total, per=per))
        
        metric_dist = single_source_dijkstra_path_length(G, source=u, weight_function=weight_function, disjunction=disjunction)
        neighbors = list(G.neighbors(u))        
        
        for v in neighbors:
            if metric_dist[v] < G[u][v][weight]:
                G.remove_edge(u, v)
    
    if distortion:
        svals = _compute_distortions(D, G, weight=weight, disjunction=disjunction)         
        return G, svals
    else:
        return G

def flagged_backbone(D, weight='weight', disjunction=sum, distortion=False, self_loops=False, *args, **kwargs):
    """
    Iterative backbone computation where edges are flagged as belonging to the backbone if they are part of an indirect shortest-path.

    Parameters
    ----------
    D : NetworkX graph
        The Distance graph
    weight : str, optional
        Edge property containing distance values, by default 'weight'
    disjunction: function (default=sum)
        Whether to sum paths or use the max value.
        Use `sum` for metric and `max` for ultrametric.
    distortion : bool, optional
        If one wants to track semi-triangular distortion, by default False
    self_loops : bool, optional
        If the distance graph has nodes with self distance greater than zero, by default False

    Returns
    -------
    NetworkX graph
        The backbone subgraph.

    Raises
    ------
    NotImplementedError
        Self-loop closure and finite step (cutoff) not implemented yet
    
    """

    if self_loops:
        raise NotImplementedError
    
    G = D.copy()
    weight_function = _weight_function(G, weight)

    edges_analyzed = []
    sorted_edges = sorted(G.edges(data=weight), key= lambda x: x[2], reverse=True)

    for u, v, _ in sorted_edges:
        if (u, v) in edges_analyzed:
            continue
        else:
            edges_analyzed.append((u, v))
            dist, path = single_source_target_dijkstra_path(G=G, source=u, target=v, 
                                                                   weight_function=weight_function, disjunction=disjunction)
            if len(path) > 2:
                
                for i in range(len(path)-1):
                    edges_analyzed.append((path[i], path[i+1]))
                
                G.remove_edge(u, v)
    
    if distortion:
        svals = _compute_distortions(D, weight=weight, disjunction=disjunction, distortion=distortion, *args, **kwargs)
        return G, svals
    else:
        return G


def _compute_distortions(D, B, weight='weight', disjunction=sum):
    """
    COMPUTE DISTORTIONS: UPDATE README
    """
    G = D.copy()
    
    G.remove_edges_from(B.edges())
    weight_function = _weight_function(B, weight)

    svals = dict()        
    for u in G.nodes():
        metric_dist = single_source_dijkstra_path_length(B, source=u, weight_function=weight_function, disjunction=disjunction)
        for v in G.neighbors(u):
            svals[(u, v)] = G[u][v][weight]/metric_dist[v]
    
    return svals   
    
    

def _check_for_kind(kind):
    """
    Check for available metric functions.
    """
    if kind not in __kinds__:
        raise TypeError("Metric not found for this algorithm. Try 'metric' or 'ultrametric',")
