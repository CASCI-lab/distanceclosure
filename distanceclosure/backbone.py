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
from itertools import product
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


def metric_backbone(
        D: nx.Graph | nx.DiGraph, 
        weight: str = 'weight', 
        distortion: bool = False, 
        self_loops: bool = False, 
        cutoff: int = None, 
        verbose: bool = False, *
        args, 
        **kwargs
    ) -> nx.Graph | nx.DiGraph:
    """ 
    
    Alias for :func:`iterative_backbone` with kind=metric.
    
    """
    
    return iterative_backbone(D, weight=weight, kind='metric', distortion=distortion, self_loops=self_loops, cutoff=cutoff, verbose=verbose, *args, **kwargs)


def ultrametric_backbone(
        D: nx.Graph | nx.DiGraph, 
        weight: str = 'weight', 
        distortion: bool = False, 
        self_loops: bool = False, 
        cutoff: int = None, 
        verbose: bool = False, 
        *args, 
        **kwargs
    ) -> nx.Graph | nx.DiGraph:
    """ 
    
    Alias for :func:`iterative_backbone`  with kind=ultrametric.
    
    """
    
    return iterative_backbone(D, weight=weight, kind='ultrametric', distortion=distortion, self_loops=self_loops, cutoff=cutoff, verbose=verbose, *args, **kwargs)


def backbone_from_closure(
        D: nx.Graph | nx.DiGraph, 
        weight: str = 'weight', 
        kind: str = 'metric', 
        distortion: bool = False, 
        self_loops: bool = False, 
        cutoff: int = None, 
        verbose: bool = False, 
        *args, 
        **kwargs
    ):
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


def iterative_backbone(
        D: nx.Graph | nx.DiGraph, 
        weight: str = 'weight', 
        kind: str = 'metric', 
        distortion: bool = False, 
        self_loops: bool = False, 
        cutoff: int = None, 
        verbose: bool = False, 
        *args, 
        **kwargs
    ) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict] :
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


def flagged_backbone(
        D: nx.Graph | nx.DiGraph, 
        weight: str = 'weight', 
        kind: str = 'metric', 
        distortion: bool = False, 
        self_loops: bool = False, 
        cutoff: int = None, 
        verbose: bool = False, 
        *args, 
        **kwargs
    ) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
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
        approx: bool = False,
        *args, **kwargs
    ) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
    """
    Heuristic backbone computation combining triangle search (based on "V. Kalavri et al (2016) Proceedings of the VLDB Endowment, Volume 9, Issue 9")

    Parameters
    ----------
    D : Directed or undirected NetworkX graph
        The weighted distance graph
    weight : str, optional
        Edge property containing distance values, by default 'weight'
    kind : str, optional
        Distance accumulation kind. Either metric (sum) or ultrametric (max), by default 'metric'
    distortion : bool, optional
        Whether to compute edge distortion from edges not in backbone, by default False
    self_loops : bool, optional
        If the distance graph has nodes with self distance greater than zero, by default False
    cutoff : int, optional
        Maximum number of connections in the path. If None, compute the entire closure as is the cutoff is the number of nodes, by default None
    approx : bool, optional
        Approximates the backbone

    Returns
    -------
    Directed or undirected NetworkX graph
        The backbone subgraph

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
        disjunction = drastic_disjunction
       
    G = D.copy()

    # Algorithm 1, page 676
    G = _local_semi_triangles(G, weight=weight, disjunction=disjunction)
    if approx:
        return G

    # Algorithm 2, page 677
    backbone_edges = _local_triangular_edges(G, weight=weight, disjunction=disjunction)

    metric_backbone = {(source, target) for source, target, _ in backbone_edges}
    unlabeled_edges = [(source, target) for source, target in G.edges() if (source, target) not in metric_backbone]


    # Algorithm 3, page 677
    remaining_metric_edges = []
    for source, target in unlabeled_edges:
        
        path = single_source_target_dijkstra_path(
            G, 
            source=source, 
            target=target, 
            weight=weight, 
            disjunction=disjunction
        )
        
        path_weights = [G[path[idx-1]][path[idx]][weight] for idx in range(1, len(path))]
        shortest_path_length = disjunction(path_weights)

        if G[source][target][weight] <= shortest_path_length:
            remaining_metric_edges.append((source, target))
    
    final_edges = list(metric_backbone) + remaining_metric_edges
    G = G.edge_subgraph(final_edges).copy()

    # Compute Distortion
    if distortion:
        svals = _compute_distortions(
            G, 
            weight=weight, 
            disjunction=disjunction, 
            distortion=distortion, 
            *args, 
            **kwargs
        )
        return G, svals
    
    return G


def drastic_disjunction(iterable: list[float]) -> float:
    iterable.sort()
    if iterable[0] == 0.0:
        return iterable[1]
    else:
        return np.inf   


def _local_semi_triangles(graph: nx.Graph | nx.DiGraph, weight: str, disjunction: Callable) -> nx.Graph | nx.DiGraph:
    for a in graph.nodes():
        neighbors = list(graph[a])
        triangles_to_check = product(neighbors, neighbors)
        for b, c in triangles_to_check:
            if graph.has_edge(a, c) and graph.has_edge(c, b) and graph.has_edge(a, b):
                ac = graph[a][c][weight]
                cb = graph[c][b][weight]
                ab = graph[a][b][weight]
                if disjunction([ cb, ac ]) < ab:
                    graph.remove_edge(a, b)
    return graph


def _local_triangular_edges(graph: nx.Graph | nx.DiGraph, weight: str, disjunction: Callable) -> nx.Graph | nx.DiGraph:
    U = {}
    for source in graph.nodes():
        neighbors = [(source, target, data[weight]) for target, data in graph[source].items()]
        U[source] = sorted(neighbors, key=lambda item: item[2])

    metric_edges = set()
    for source in graph.nodes():
        if not U[source]:
            continue

        weights_for_comparison = set()
        metric = True

        removed_pair = U[source].pop(0)
        metric_edges.add(removed_pair)
        
        while U[source]:
            e = U[source].pop(0)
            for _, target, _ in metric_edges:
                if graph.has_edge(source, target) and U[target]:
                    w_x = disjunction([graph[source][target][weight], U[target][0][2]])
                    weights_for_comparison.add(w_x)
            
            for w in weights_for_comparison:
                if e[2] > w:
                    metric = False
                    break

            if metric:
                metric_edges.add(e)
                weights_for_comparison = set()
            else:
                return metric_edges


def _compute_distortions(
        D: nx.Graph | nx.DiGraph, 
        B: nx.Graph | nx.DiGraph, 
        weight: str = 'weight', 
        disjunction: Callable = sum
    ) -> dict:
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


def _check_for_kind(kind: str) -> None:
    """
    Check for available metric functions.
    """
    if kind not in __kinds__:
        raise TypeError("Metric not found for this algorithm. Try 'metric' or 'ultrametric',")

