# -*- coding: utf-8 -*-
"""
Backbone Subgraph - Fast Implementation
==================

Computes the shortest path distance backbone on a weighted graph.
These algorithms work with edges weighted as distances.
"""

import numpy as np
import networkx as nx
from distanceclosure.dijkstra import single_source_dijkstra_path_length
from networkx.algorithms.shortest_paths.weighted import _weight_function

__name__ = 'distanceclosure'
__author__ = """\n""".join(['Rion Brattig Correia <rionbr@gmail.com>', 'Felipe Xavier Costa <fcosta@binghamton.com>'])

__all__ = [
    "backbone",
    "metric_backbone",
    "ultrametric_backbone"
]


__kinds__ = ['metric', 'ultrametric']
__algorithms__ = ['dense', 'dijkstra']

def metric_backbone(D, weight='weight', distortion=False, self_loops=False, cutoff=None, verbose=False, *args, **kwargs):
    """ Alias for :func:`backbone` with kind=metric.
    """
    
    return backbone(D, weight=weight, kind='metric', distortion=distortion, self_loops=self_loops, cutoff=cutoff, verbose=verbose, *args, **kwargs)

def ultrametric_backbone(D, weight='weight', distortion=False, self_loops=False, cutoff=None, verbose=False, *args, **kwargs):
    """ Alias for :func:`backbone`  with kind=ultrametric.
    """
    
    return backbone(D, weight=weight, kind='ultrametric', distortion=distortion, self_loops=self_loops, cutoff=cutoff, verbose=verbose, *args, **kwargs)

def backbone(D, weight='weight', kind='metric', distortion=False, self_loops=False, cutoff=None, verbose=False, *args, **kwargs):
    """
    Fast backbone (only) computation considering node ordering.

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
        G, s_values = _compute_backbone(D, weight=weight, disjunction=sum, distortion=distortion, verbose=verbose, *args, **kwargs)
    elif kind == 'ultrametric':
        G, s_values = _compute_backbone(D, weight=weight, disjunction=max, distortion=distortion, verbose=verbose, *args, **kwargs)
    
    if distortion:
        return G, s_values
    else:
        return G


def _compute_backbone(D, weight='weight', disjunction=sum, distortion=False, self_loops=False, verbose=False, *args, **kwargs):
    """
    Fast backbone (only) computation considering node ordering.

    Parameters
    ----------
    D : NetworkX graph
        The Distance graph
    weight : str, optional
        Edge property containing distance values, by default 'weight'
    disjunction: function (default=sum)
        Whether to sum paths or use the max value.
        Use `sum` for metric and `max` for ultrametric.
    self_loops : bool, optional
        If the distance graph has nodes with self distance greater than zero, by default False
    distortion : bool, optional
        If one wants to track semi-triangular distortion, by default False
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
    
    G = D.copy()

    ordered_nodes = sorted(G.degree(weight=weight), key=lambda x: x[1], reverse=True)

    weight_function = _weight_function(G, weight)
    
    if verbose:
        total = G.number_of_nodes()
        i = 0
        
    s_values = dict()

    for n, _ in ordered_nodes:
        if verbose:
            i += 1
            per = i/total
            print("Closure: Dijkstra : source node {u:s} : {i:d} of {total:d} ({per:.2%})".format(u=n, i=i, total=total, per=per))

        metric_dist = single_source_dijkstra_path_length(G, source=n, weight_function=weight_function, disjunction=disjunction)
        neighbors = list(G.neighbors(n)) # Need to be separate or will raise changing list error
        for v in neighbors:
            if metric_dist[v] < G[n][v][weight]:
                if distortion:
                    s_values[(n, v)] = G[n][v][weight]/metric_dist[v]
                G.remove_edge(n, v)
    
    return G, s_values


def _check_for_kind(kind):
    """
    Check for available metric functions.
    """
    if kind not in __kinds__:
        raise TypeError("Metric not found for this algorithm. Try 'metric' or 'ultrametric',")
