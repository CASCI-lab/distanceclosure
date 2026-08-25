# -*- coding: utf-8 -*-
"""
Distance Backbone
=================

Compute the distance backbones of both directed and undirected weighted graphs.
"""
import numpy as np
import networkx as nx

from distanceclosure._dijkstra import _single_source_dijkstra_path_length, _single_source_target_dijkstra_path, _single_source_neighbors_dijkstra_path_length
from distanceclosure.closure import distance_closure
from distanceclosure._constants import _KINDS

from itertools import product
from typing import Callable

__all__ = [
    "distance_backbone",
    "metric_backbone",
    "ultrametric_backbone" 
]


# Public
def distance_backbone(D: nx.Graph | nx.DiGraph, weight: str = "weight", kind: str = "metric", algorithm: str = "iterative", distortion: bool = False, self_loops: bool = False, cutoff: int = None, verbose: bool = False) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
    """
    Compute the distance backbone of a weighted graph.

    Parameters
    ----------
    D : Directed or undirected NetworkX graph
        A weighted distance graph
    weight : str, optional
        Edge property containing distance values, by default 'weight'
    kind : {"metric", "ultrametric"}, optional
        Distance metric used to compute the backbone. "metric" uses sum and "ultrametric" uses max, by default "metric".
    algorithm : {"iterative", "flagged", "closure", "heuristic", "approximate"}, optional
        Algorithm used to compute the backbone, by default "iterative".
    distortion : bool, optional
        Whether to compute edge distortion from edges not in backbone, by default False
    self_loops : bool, optional
        If the distance graph has nodes with self distance greater than zero, by default False
    cutoff : int, optional
        Maximum number of connections in the path. If None, compute the entire closure as is the cutoff is the number of nodes, by default None
    verbose : bool, optional
        Whether to display progress information, by default False.

    Returns
    -------
    nx.Graph or nx.DiGraph
        The distance backbone.
    dict
        Edge distortions, returned with the backbone when ``distortion=True``.

    Raises
    ------
    ValueError
        If ``kind`` or ``algorithm`` is invalid.
    """

    if self_loops:
        raise NotImplementedError
    elif cutoff is not None:
        raise NotImplementedError
    
    if kind == 'metric':
        disjunction = sum
    elif kind == 'ultrametric':
        disjunction = max
    elif kind == 'drastic':
        disjunction = _drastic_disjunction

    if kind not in _KINDS:
        raise ValueError("Invalid input. Valid arguments are 'metric' and 'ultrametric'.")

    try:
        chosen_algorithm = _BACKBONE_ALGORITHMS[algorithm]
    except KeyError:
        raise ValueError("Invalid input. Valid arguments are 'iterative', 'flagged', 'closure', 'heuristic', or 'approximate'")
    
    return chosen_algorithm(D, weight=weight, disjunction=disjunction, distortion=distortion, self_loops=self_loops, cutoff=cutoff, verbose=verbose)


def metric_backbone(D: nx.Graph | nx.DiGraph, weight: str = "weight", distortion: bool = False, self_loops: bool = False, cutoff: int = None, verbose: bool = False) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
    """
    Compute the metric backbone of a weighted graph.

    This is a wrapper for :func:`distance_backbone`
    where ``kind="metric"`` and ``algorithm="iterative"``.
    """

    return distance_backbone(D, weight=weight, algorithm="iterative", kind="metric", distortion=distortion, self_loops=self_loops, cutoff=cutoff, verbose=verbose)


def ultrametric_backbone(D: nx.Graph | nx.DiGraph, weight: str = "weight", distortion: bool = False, self_loops: bool = False, cutoff: int = None, verbose: bool = False) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
    """
    Compute the metric backbone of a weighted graph.

    This is a wrapper for :func:`distance_backbone`
    where ``kind="ultrametric"`` and ``algorithm="iterative"``.
    """

    return distance_backbone(D, weight=weight, algorithm="iterative", kind="ultrametric", distortion=distortion, self_loops=self_loops, cutoff=cutoff, verbose=verbose)


# Private 
def _flagged_backbone(D: nx.Graph | nx.DiGraph, weight: str, disjunction: Callable, distortion: bool, self_loops: bool, cutoff: int, verbose: bool) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
    G = D.copy()
    B = nx.DiGraph() if nx.is_directed(G) else nx.Graph()

    if verbose: 
        total = G.number_of_nodes()
        i = 0

    for node in list(G.nodes()):
        shortest_paths_to_neighbors = _single_source_neighbors_dijkstra_path_length(G, source=node, weight=weight, disjunction=disjunction)

        for neighbor in list(G.neighbors(node)):
            shortest_path = shortest_paths_to_neighbors[neighbor]
            direct_path = G[node][neighbor][weight]

            if shortest_path < direct_path:
                G.remove_edge(node, neighbor)
            else:
                B.add_edge(node, neighbor)

        if B.number_of_edges() == G.number_of_edges():
            break    

        if verbose:
            i += 1
            per = i / total
            print("Flagged Backbone : dijkstra : {disjunction:s} : {i:d} of {total:d} ({per:.2%})".format(i=i, total=total, per=per, disjunction=disjunction.__name__))
    
    if distortion:
        svals = _compute_distortions(D, G, weight=weight, disjunction=disjunction)
        return G, svals

    return G

    
def _iterative_backbone(D: nx.Graph | nx.DiGraph, weight: str, disjunction: Callable, distortion: bool, self_loops: bool , cutoff: int, verbose: bool) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
    G = D.copy()
    
    if verbose:
        total = G.number_of_nodes()
        i = 0
    
    for node in list(G.nodes()):
        shortest_paths_to_neighbors = _single_source_neighbors_dijkstra_path_length(G, source=node, weight=weight, disjunction=disjunction)

        for neighbor in list(G.neighbors(node)):
            shortest_path = shortest_paths_to_neighbors[neighbor]
            direct_path = G[node][neighbor][weight]

            if shortest_path < direct_path:
                G.remove_edge(node, neighbor)

        if verbose:
            i += 1
            per = i/total
            print("Iterative Backbone : dijkstra : {disjunction:s} : {i:d} of {total:d} ({per:.2%})".format(i=i, total=total, per=per, disjunction=disjunction.__name__))
     
    if distortion:
        svals = _compute_distortions(D, G, weight=weight, disjunction=disjunction)    
        return G, svals

    return G


def _closure_backbone(D: nx.Graph | nx.DiGraph, weight: str, disjunction: Callable, distortion: bool, self_loops: bool, cutoff: int, verbose: bool) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
    # Come back and fix this once I am done with the closure
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

    if disjunction == sum:
        kind = "metric"
    elif disjunction == max:
        kind = "ultrametric"
    elif disjunction == _drastic_disjunction:
        kind = "drastic"

    DC = distance_closure(D, kind=kind, algorithm='dijkstra', weight=weight, existing_edges_only=True, verbose=verbose)

    is_kind = 'is_{kind:s}'.format(kind=kind)
    metric_edges = [(u, v) for u, v in DC.edges() if DC[u][v][is_kind]]
    G = DC.edge_subgraph(metric_edges).copy()
    
    if distortion:
        svals = _compute_distortions(D, G, weight=weight, kind=kind)         
        return G, svals

    return G


def _heuristic_backbone(D: nx.Graph | nx.DiGraph, weight: str, disjunction: Callable, distortion: bool, self_loops: bool, cutoff: int, verbose: bool) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
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

    G = D.copy()

    # Algorithm 1, page 676
    G = _local_semi_triangles(G, disjunction=disjunction, weight=weight)

    # Algorithm 2, page 677
    backbone_edges = _local_triangular_edges(G, disjunction=disjunction, weight=weight)

    metric_backbone = {(source, target) for source, target, _ in backbone_edges}
    unlabeled_edges = [(source, target) for source, target in G.edges() if (source, target) not in metric_backbone]


    # Algorithm 3, page 677
    remaining_metric_edges = []
    for source, target in unlabeled_edges:
        
        path = _single_source_target_dijkstra_path(G, source=source, target=target, weight=weight, disjunction=disjunction)
        
        path_weights = [G[path[idx-1]][path[idx]][weight] for idx in range(1, len(path))]
        shortest_path_length = disjunction(path_weights)

        if G[source][target][weight] <= shortest_path_length:
            remaining_metric_edges.append((source, target))
    
    final_edges = list(metric_backbone) + remaining_metric_edges
    G = G.edge_subgraph(final_edges).copy()

    # Compute Distortion
    if distortion:
        svals = _compute_distortions(D, G, weight=weight, disjunction=disjunction)
        return G, svals
    
    return G


def _approximate_backbone(D: nx.Graph | nx.DiGraph, weight: str, disjunction: Callable, distortion: bool, self_loops: bool, cutoff: int) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
    G = D.copy()

    # Algorithm 1, page 676
    G = _local_semi_triangles(G, disjunction=disjunction, weight=weight)

    # Compute Distortion
    if distortion:
        svals = _compute_distortions(D, G, weight=weight, disjunction=disjunction)
        return G, svals
    
    return G


def _drastic_disjunction(iterable: list[float]) -> float:
    iterable.sort()
    if iterable[0] == 0.0:
        return iterable[1]
    else:
        return np.inf   


def _local_semi_triangles(graph: nx.Graph | nx.DiGraph, disjunction: Callable, weight: str = 'weight') -> nx.Graph | nx.DiGraph:
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


def _local_triangular_edges(graph: nx.Graph | nx.DiGraph, disjunction: Callable, weight: str = 'weight') -> nx.Graph | nx.DiGraph:
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
            
    return metric_edges


def _compute_distortions(D: nx.Graph | nx.DiGraph, B: nx.Graph | nx.DiGraph, disjunction: Callable, weight: str) -> dict:
    """
    Compute distortions of edges not in backbone.

    Parameters
    ----------
    D : Directed or undirected NetworkX distance graph
        The weighted distance graph
    B : Directed or undirected NetworkX backbone graph
        The weighted backbone subgraph
    weight : str, optional
        Edge property containing distance values, by default 'weight'
    kind : str, optional
        Distance accumulation kind. Either metric (sum) or ultrametric (max), by default 'metric'

    Returns
    -------
    Dictionary keyed by edge with its distortion value.
    
    """
    G = D.copy()
    G.remove_edges_from(B.edges())

    svals = dict()        
    for u in G.nodes():
        metric_dist = _single_source_dijkstra_path_length(B, source=u, weight="weight", disjunction=disjunction)
        
        for v in G.neighbors(u):
            svals[(u, v)] = G[u][v][weight]/metric_dist[v]
    
    return svals   

_BACKBONE_ALGORITHMS = {
    "iterative": _iterative_backbone,
    "flagged": _flagged_backbone,
    "closure": _closure_backbone,
    "heuristic": _heuristic_backbone,
    "approximate": _approximate_backbone
}
