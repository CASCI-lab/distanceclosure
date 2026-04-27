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
from distanceclosure.closure import distance_closure
from networkx.algorithms.shortest_paths.weighted import _weight_function

__name__ = 'distanceclosure'
__author__ = """\n""".join(['Rion Brattig Correia <rionbr@gmail.com>', 'Felipe Xavier Costa <fcosta@binghamton.com>'])

__all__ = [
    "metric_backbone",
    "ultrametric_backbone",
    "iterative_backbone",
    "flagged_backbone",
    "backbone_from_closure",
    "heuristic_undirected_backbone"
]

__kinds__ = ['metric', 'ultrametric', 'drastic']
__algorithms__ = ['dense', 'dijkstra']

def metric_backbone(D, weight='weight', distortion=False, self_loops=False, cutoff=None, verbose=False, *args, **kwargs):
    """ Alias for :func:`iterative_backbone` with kind=metric.
    """
    
    return iterative_backbone(D, weight=weight, kind='metric', distortion=distortion, self_loops=self_loops, cutoff=cutoff, verbose=verbose, *args, **kwargs)


def ultrametric_backbone(D, weight='weight', distortion=False, self_loops=False, cutoff=None, verbose=False, *args, **kwargs):
    """ Alias for :func:`iterative_backbone`  with kind=ultrametric.
    """
    
    return iterative_backbone(D, weight=weight, kind='ultrametric', distortion=distortion, self_loops=self_loops, cutoff=cutoff, verbose=verbose, *args, **kwargs)


def backbone_from_closure(D, weight='weight', kind='metric', distortion=False, self_loops=False, cutoff=None, verbose=False, *args, **kwargs):
    """
    Backbone computation considering the closure.

    Parameters
    ----------
    D : NetworkX graph
        The Distance graph
    weight : str, optional
        Edge property containing distance values, by default 'weight'
    kind : str, optional
        Distance accumulation kind. Either metric (sum) or ultrametric (max), by default 'metric'
    distortion : bool, optional
        Whether to compute edge distortion from edges not in backbone, by default False
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

    if self_loops:
        raise NotImplementedError
    if cutoff is not None:
        raise NotImplementedError

    DC = distance_closure(D, kind=kind, algorithm='dijkstra', weight=weight, only_backbone=True, verbose=verbose, *args, **kwargs)
    is_kind = 'is_{kind:s}'.format(kind=kind)
    metric_edges = [(u, v) for u, v in DC.edges() if DC[u][v][is_kind]]
    G = DC.edge_subgraph(metric_edges).copy()
    
    if distortion:
        svals = _compute_distortions(D, G, weight=weight, disjunction=disjunction)         
        return G, svals
    else:
        return G


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
    distortion : bool, optional
        Whether to compute edge distortion from edges not in backbone, by default False
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
    elif kind == 'drastic':
        disjunction=drastic_disjunction
    
    G = D.copy()
    weight_function = _weight_function(G, weight)
    
    if verbose:
        total = G.number_of_nodes()
        i = 0
    
    for u, _ in sorted(G.degree(weight=weight), key=lambda x: x[1]):
        if verbose:
            i += 1
            per = i/total
            print("Iterative Backbone : dijkstra : {kind:s} : {i:d} of {total:d} ({per:.2%})".format(i=i, total=total, per=per, kind=kind))
        
        metric_dist = single_source_dijkstra_path_length(G, source=u, weight_function=weight_function, disjunction=disjunction)
        for v in list(G.neighbors(u)):
            if metric_dist[v] < G[u][v][weight]:
                G.remove_edge(u, v)
    
    if distortion:
        svals = _compute_distortions(D, G, weight=weight, disjunction=disjunction)         
        return G, svals
    else:
        return G


def flagged_backbone(D, weight='weight', kind='metric', distortion=False, self_loops=False, cutoff=None, verbose=False, *args, **kwargs):
    """

    Iterative backbone computation where edges are flagged as belonging to the backbone if they are part of an indirect shortest-path.

    Parameters
    ----------
    D : NetworkX graph
        The Distance graph
    weight : str, optional
        Edge property containing distance values, by default 'weight'
    kind : str, optional
        Distance accumulation kind. Either metric (sum) or ultrametric (max), by default 'metric'
    distortion : bool, optional
        Whether to compute edge distortion from edges not in backbone, by default False
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
    elif kind == 'drastic':
        disjunction=drastic_disjunction
        
    G = D.copy()
    weight_function = _weight_function(G, weight)

    B = nx.DiGraph() if nx.is_directed(G) else nx.Graph()

    if verbose: 
        total = G.number_of_nodes()
        i = 0

    for u, _ in sorted(G.degree(weight=weight), key=lambda x: x[1]):
        if verbose:
            i += 1
            per = i/total
            print("Flagged Backbone : dijkstra : {kind:s} : {i:d} of {total:d} ({per:.2%})".format(i=i, total=total, per=per, kind=kind))

        metric_dist = single_source_dijkstra_path_length(G, source=u, weight_function=weight_function, disjunction=disjunction)
        for v in list(G.neighbors(u)):
            if metric_dist[v] < G[u][v][weight]:
                G.remove_edge(u, v)
            else:
                B.add_edge(u, v)

        if B.number_of_edges() == G.number_of_edges():
            break    
    
    if distortion:
        svals = _compute_distortions(D, G, weight=weight, disjunction=disjunction)
        return G, svals
    else:
        return G
    

def heuristic_undirected_backbone(D, weight='weight', kind='metric', distortion=False, self_loops=False, cutoff=None, verbose=False, *args, **kwargs):
    """
    Heuristic backbone computation combining triangle search (based on "V. Kalavri et al (2016) Proceedings of the VLDB Endowment, Volume 9, Issue 9") with :func:`iterative_backbone`.

    Parameters
    ----------
    
    D : NetworkX graph
        The Distance graph
    weight : str, optional
        Edge property containing distance values, by default 'weight'
    kind : str, optional
        Distance accumulation kind. Either metric (sum) or ultrametric (max), by default 'metric'
    distortion : bool, optional
        Whether to compute edge distortion from edges not in backbone, by default False
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
    
    from itertools import pairwise
    
    _check_for_kind(kind)
    
    if self_loops:
        raise NotImplementedError
    if cutoff is not None:
        raise NotImplementedError
    if nx.is_directed(D):
        raise NotImplementedError
    
    if kind == 'metric':
        disjunction = sum
    elif kind == 'ultrametric':
        disjunction = max
    elif kind == 'drastic':
        disjunction=drastic_disjunction
        
    G = D.copy()
    
    #print('Semi-metric Triangles')
    for v in G.nodes():
        possible_triangles = list(pairwise(G[v]))
        for x, y in possible_triangles:
            if G.has_edge(x, y):
                if disjunction([G[x][y][weight], G[y][v][weight]]) < G[x][v][weight]:
                    G.remove_edge(x, v)
    
    #print('Local Metric') 
    metric_edges = set()
    U = {v: [(x, d[weight]) for x, d in sorted(G[v].items(), key=lambda item: item[1][weight])] for v in G.nodes()} 
    for v in G.nodes():
        W = set()
        metric = True
        metric_edges.add((v, U[v].pop(0)[0]))
        
        while len(U[v]) > 0:
            e = U[v].pop(0)
            for _, x in metric_edges:
                if G.has_edge(v, x) and len(U[x])>0:
                    wx = disjunction([G[v][x][weight], U[x][0][1]])
                    W.add(wx)
                
            for w in W:
                if e[1] > w:
                    metric = False
                    break
            
            if metric:
                metric_edges.add((v, e[0]))
                W = set()
            else:
                continue
    
    unlabeled_edges = [(u, v) for u, v in G.edges() if (u, v) not in metric_edges]

    for u, v in unlabeled_edges:
        Pu = single_source_target_dijkstra_path(G, source=u, target=v, weight=weight, disjunction=disjunction)
        spl = disjunction([G[Pu[idx-1]][Pu[idx]][weight] for idx in range(1, len(Pu))])
        if G[u][v][weight] <= spl:
            metric_edges.add((u, v))

    G = G.edge_subgraph(metric_edges).copy()
    G = iterative_backbone(G, weight=weight, kind=kind, distortion=distortion)

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


def drastic_disjunction(iterable):
        
    iterable.sort()
    if iterable[0] == 0.0:
        return iterable[1]
    else:
        return np.inf
    

def _check_for_kind(kind):
    """
    Check for available metric functions.
    """
    if kind not in __kinds__:
        raise TypeError("Metric not found for this algorithm. Try 'metric' or 'ultrametric',")
