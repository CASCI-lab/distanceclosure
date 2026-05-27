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
from itertools import permutations, pairwise
from collections.abc import Callable

__name__ = 'distanceclosure'
__author__ = """\n""".join(['Rion Brattig Correia <rionbr@gmail.com>', 'Felipe Xavier Costa <fcosta@binghamton.com>'])

__all__ = [
    "metric_backbone",
    "ultrametric_backbone",
    "iterative_backbone",
    "flagged_backbone",
    "backbone_from_closure",
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
        svals = _compute_distortions(D, G, weight=weight, disjunction=drastic_disjunction)         
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
    

def heuristic_backbone(
        D: nx.Graph | nx.DiGraph, 
        weight: str = 'weight', 
        kind: str = 'metric', 
        distortion: bool = False, 
        self_loops: bool = False, 
        cutoff: int = None, 
        *args, **kwargs
    ) -> nx.Graph | nx.DiGraph:
    """
    Heuristic backbone computation combining triangle search (based on "V. Kalavri et al (2016) Proceedings of the VLDB Endowment, Volume 9, Issue 9")

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
    
    if kind == 'metric':
        disjunction = sum
    elif kind == 'ultrametric':
        disjunction = max
    elif kind == 'drastic':
        disjunction=drastic_disjunction
        
    G = D.copy()
    
    if G.is_directed():
        G = _algorithm_one_directed(G, weight=weight, disjunction=disjunction)
    else:
        G = _algorithm_one_undirected(G, weight=weight, disjunction=disjunction)

    metric_edges = _algorithm_two(G, weight=weight, disjunction=disjunction)

    if G.is_directed():
        metric_pairs = {(s, t) for s, t, _ in metric_edges}
        unlabeled_edges = [(s, t) for s, t in G.edges() if (s, t) not in metric_pairs]
    else:
        metric_pairs = {_uniform_edge(s, t) for s, t, _ in metric_edges}
        unlabeled_edges = [(s, t) for s, t in G.edges() if _uniform_edge(s, t) not in metric_pairs]

    more_metric_edges = _algorithm_three(G, unlabeled_edges=unlabeled_edges, weight=weight, disjunction=disjunction)
    final_edges = list(metric_pairs) + more_metric_edges

    G = G.edge_subgraph(final_edges).copy()

    if distortion:
        svals = _compute_distortions(D, weight=weight, disjunction=disjunction, distortion=distortion, *args, **kwargs)
        return G, svals
    else:
        return G


def drastic_disjunction(iterable):
        
    iterable.sort()
    if iterable[0] == 0.0:
        return iterable[1]
    else:
        return np.inf   


def _algorithm_one_directed(graph: nx.DiGraph, weight: str, disjunction: Callable) -> nx.DiGraph:
    for a in graph.nodes():
        triangles_to_check = list(permutations(graph[a], 2)) 
        for b, c in triangles_to_check:
           if graph.has_edge(a, b) and graph.has_edge(b, c) and graph.has_edge(c, a):
            bc = graph[b][c][weight]
            ca = graph[c][a][weight]
            ab = graph[a][b][weight]
            if disjunction([ bc, ca ]) < ab:
                    graph.remove_edge(a, b)
    return graph


def _algorithm_one_undirected(graph: nx.Graph, weight: str, disjunction: Callable) -> nx.Graph:
    for a in graph.nodes():
        triangles_to_check = list(pairwise(graph[a])) 
        for b, c in triangles_to_check:
           if graph.has_edge(b, c):
            bc = graph[b][c][weight]
            ca = graph[c][a][weight]
            ba = graph[b][a][weight]
            if disjunction([ bc, ca ]) < ba:
                    graph.remove_edge(b, a)
    return graph


def _algorithm_two(graph: nx.Graph | nx.DiGraph, weight: str, disjunction: Callable) -> nx.Graph | nx.DiGraph:
    U = {}
    for s in graph.nodes():
        neighbors = [(s, t, data[weight]) for t, data in graph[s].items()]
        U[s] = sorted(neighbors, key=lambda item: item[2])

    metric_edges = set()
    for s in graph.nodes():
        if not U[s]:
            continue

        weights_for_comparison = set()
        metric = True

        removed_pair = U[s].pop(0)
        metric_edges.add(removed_pair)
        
        while U[s]:
            e = U[s].pop(0)
            for _, target, _ in metric_edges:
                if graph.has_edge(s, target) and U[target]:
                    w_x = disjunction([graph[s][target][weight], U[target][0][2]])
                    weights_for_comparison.add(w_x)
            
            for w in weights_for_comparison:
                if e[2] > w:
                    metric = False
                    break

            if metric:
                metric_edges.add(e)
                weights_for_comparison = set()
            else:
                break
    
    return metric_edges


def _algorithm_three(graph: nx.Graph | nx.DiGraph, unlabeled_edges: list, weight: str, disjunction: Callable) -> list:
    more_metric_edges = []
    for s, t in unlabeled_edges:
        Pu = single_source_target_dijkstra_path(
            graph, 
            source=s, 
            target=t, 
            weight=weight, 
            disjunction=disjunction
        )
        
        spl = disjunction([graph[Pu[idx-1]][Pu[idx]][weight] for idx in range(1, len(Pu))])
        if graph[s][t][weight] <= spl:
            more_metric_edges.append((s, t))
    
    return more_metric_edges


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


def _uniform_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)
