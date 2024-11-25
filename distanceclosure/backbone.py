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
    "flagged_backbone",
    "heuristic_backbone"
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
    elif kind == 'drastic':
        disjunction=drastic_disjunction
    
    G = D.copy()
    weight_function = _weight_function(G, weight)
    
    sorted_edges = sorted(G.edges(data=weight), key= lambda x: x[2], reverse=True)

    if verbose:
        total = G.number_of_edges()
        i = 0
    
    for u, v, dist in sorted_edges:
        if verbose:
            i += 1
            per = i/total
            print("Backbone: Iterative Dijkstra: {i:d} of {total:d} ({per:.2%})".format(i=i, total=total, per=per))
        
        metric_dist = single_source_dijkstra_path_length(G, source=u, weight_function=weight_function, disjunction=disjunction)        
        if metric_dist[v] < dist:
            G.remove_edge(u, v)
            
    if distortion:
        svals = _compute_distortions(D, weight=weight, disjunction=disjunction, distortion=distortion, verbose=verbose, *args, **kwargs)         
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

    edges_analyzed = []
    sorted_edges = sorted(G.edges(data=weight), key= lambda x: x[2], reverse=True)
    
    if verbose:
        total = G.number_of_edges()

    for u, v, _ in sorted_edges:
        if (u, v) in edges_analyzed:
            continue
        else:
            edges_analyzed.append((u, v))
            
            path = single_source_target_dijkstra_path(G=G, source=u, target=v, weight_function=weight_function, disjunction=disjunction)
            if len(path) > 2:
                
                for i in range(len(path)-1):
                    edges_analyzed.append((path[i], path[i+1]))
                    
                G.remove_edge(u, v)
            
            if verbose:
                per = len(edges_analyzed)/total
                print("Backbone: Flagged Dijkstra: {nedges:d} of {total:d} ({per:.2%})".format(nedges=len(edges_analyzed), total=total, per=per))            
    
    if distortion:
        svals = _compute_distortions(D, weight=weight, disjunction=disjunction, distortion=distortion, *args, **kwargs)
        return G, svals
    else:
        return G


def _compute_distortions(D, B, weight='weight', disjunction=sum, self_loops=False):
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
    
    

def heuristic_backbone(D, weight='weight', kind='metric', distortion=False, self_loops=False, cutoff=None, verbose=False, *args, **kwargs):
    """
    Heuristic backbone computation combining triangle search (based on "V. Kalavri et al (2016) Proceedings of the VLDB Endowment, Volume 9, Issue 9") with :func:`iterative_backbone`.

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

    sorted_edges = sorted(G.edges(data=weight), key= lambda x: x[2], reverse=True)
    
    for u, v, dist in sorted_edges:

        metric_dist_v = dist        
        neighbors = list(G.neighbors(u)) # Need to be separate or will raise changing list error        
        for k in neighbors:
            if G.has_edge(k, v):
                metric_dist_v = min(metric_dist_v, disjunction([G[u][k][weight], G[k][v][weight]]))

        if metric_dist_v < dist:
            G.remove_edge(u, v)
        else:
            metric_dist = single_source_dijkstra_path_length(G, source=u, weight_function=weight_function, disjunction=disjunction)            
            if metric_dist[v] < dist:
                G.remove_edge(u, v)    
    
    if distortion:
        svals = _compute_distortions(D, weight=weight, disjunction=disjunction, distortion=distortion, *args, **kwargs)
        return G, svals
    else:
        return G


def _compute_heuristic_backbone(D, weight='weight', disjunction=sum, triangles=True, verbose=False, *args, **kwargs):
    """
    Heuristic backbone computation as described in: "V. Kalavri et al (2016) Proceedings of the VLDB Endowment, Volume 9, Issue 9"

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
    
    """
    
    G = D.copy()

    ordered_nodes = sorted(G.degree(weight=weight), key=lambda x: x[1], reverse=True)

    weight_function = _weight_function(G, weight)
    
    if verbose:
        total = G.number_of_nodes()
        i = 0
    
    metric_edges = []
    edges = dict()
                
    for n, _ in ordered_nodes:
        if verbose:
            i += 1
            per = i/total
            print("Closure: Dijkstra : source node {u:s} : {i:d} of {total:d} ({per:.2%})".format(u=n, i=i, total=total, per=per))

        if triangles:
            neighbors = list(G.neighbors(n)) # Need to be separate or will raise changing list error
            for idx in range(len(neighbors)):
                for jdx in range(idx+1, len(neighbors)):
                    if G.has_edge(neighbors[idx], neighbors[jdx]):
                        distances = {(n, neighbors[idx]): G[n][neighbors[idx]][weight], (n, neighbors[jdx]): G[n][neighbors[jdx]][weight], (n, neighbors[jdx]): G[n][neighbors[jdx]][weight]}
                        distances = [(edge, dist) for edge, dist in sorted(distances.items(), key=lambda item: item[1])]
                        
                        dist = disjunction([distances[0][1], distances[1][1]])
                        if dist < distances[2][1]:
                            G.remove_edge(distances[2][0])
        
        distances = {node: G[n][node][weight] for node in G.neighbors(n)}
        edges[n] = []
        for node, _ in sorted(distances.items(), key=lambda item: item[1]):
            edges.append(node)
            
        W = []
        metric = True
        metric_edges.append((n, edges[n].pop(0)))
        
        while len(edges[n]) > 0:
            e = edges[n].pop(0)
            for m in metric_edges:
                x = m[1]
                W.append(disjunction([G[n][x][weight], G[x][edges[x].pop(0)]]))
            
            for w in W:
                if G[n][e][weight] > w:
                    metric = False
                    break
            
            if metric:
                metric_edges.append((n, e))
                W = []
            else:
                break
    
    
    g = G.copy()
    g.remove_edges_from(metric_edges)
    for u, v in g.edges():
        metric_dist = single_source_dijkstra_path_length(G, source=u, weight_function=weight_function, disjunction=disjunction)
        if metric_dist[v] < g[u][v][weight]:
            G.remove_edge((u, v))
    
    return G


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
