# -*- coding: utf-8 -*-
"""
Distance Backbone
=================

Compute the distance backbone of both directed and undirected weighted graphs.

Authors
-------
Felipe Xavier Costa <fcosta@binghamton.edu>
Robert Palermo <rpalermo@binghamton.edu>
"""

import numpy as np
import networkx as nx

from distanceclosure.dijkstra import single_source_target_dijkstra_path, single_source_neighbors_dijkstra_path_length
from distanceclosure.closure import distance_closure

from itertools import product
from collections.abc import Callable, Hashable


__all__ = [
    "distance_backbone",
    "metric_backbone",
    "ultrametric_backbone" 
]


def distance_backbone(D: nx.Graph | nx.DiGraph, weight: str = "weight", geometry: str = "metric", algorithm: str = "stlp", distortion: bool = False, self_loops: bool = False, cutoff: int = None, verbose: bool = False) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
    """
    Compute the distance backbone of either a directed or an undirected weighted graph.

    Parameters
    ----------
    D : Directed or undirected NetworkX graph
        A directed or undirected weighted distance graph.
    weight : str, optional
        Edge attribute containing distance values. ``weight="weight"`` by default.
    geometry : {"metric", "ultrametric"}, optional
        Distance metric used to compute the backbone. "metric" uses sum and "ultrametric" uses max. ``kind="metric"`` by default.
    algorithm : {"stlp", "ttlp", "closure", "ksl", "approx"}, optional
        Algorithm used to compute the backbone. ``algorithm="stlp"`` by default.
    distortion : bool, optional
        Whether to compute edge distortion for edges not in the backbone. If True, returns a dictionary of edge distortions. ``distortion=False`` by default.
    self_loops : bool, optional
        Whether to remove semi-triangular self-loops.``self_loops=False`` by default. 
    cutoff : int, optional
        Restrict the maximum number of edges a shortest path can contain. If None, compute the backbone with no restriction. ``cutoff=None`` by default.
    verbose : bool, optional
        Whether to display backbone computation progress. ``verbose=False`` by default.

    Returns
    -------
    nx.Graph or nx.DiGraph
        The distance backbone of either a directed or an undirected weighted graph.
    tuple of (nx.Graph or nx.DiGraph, dict)
        Distortions of edges not in the backbone, returned with the backbone when ``distortion=True``.

    Raises
    ------
    KeyError 
        If ``geometry`` is not a valid key in ``_GEOMETRY`` or ``algorithm`` is not a valid key in ``_BACKBONE_ALGORITHMS``.
    """
    
    try:
        chosen_td_norm = _GEOMETRY[geometry]
    except KeyError:
        raise ValueError(f"Invalid input. Valid arguments are: {list(_GEOMETRY)}") from None

    try:
        chosen_algorithm = _BACKBONE_ALGORITHMS[algorithm]
    except KeyError:
        raise ValueError(f"Invalid input. Valid arguments are: {list(_BACKBONE_ALGORITHMS)}") from None
    
    if chosen_algorithm is _BACKBONE_ALGORITHMS["closure"]:
        return chosen_algorithm(D, weight=weight, geometry=geometry, td_norm=chosen_td_norm, distortion=distortion, self_loops=self_loops, cutoff=cutoff, verbose=verbose)
    
    return chosen_algorithm(D, weight=weight, td_norm=chosen_td_norm, distortion=distortion, self_loops=self_loops, cutoff=cutoff, verbose=verbose)


def metric_backbone(D: nx.Graph | nx.DiGraph, weight: str = "weight", distortion: bool = False, self_loops: bool = False, cutoff: int = None, verbose: bool = False) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
    """
    Compute the metric backbone of either a directed or an undirected weighted graph.

    This is a wrapper for :func:`distance_backbone`
    where ``geometry="metric"`` and ``algorithm="stlp"``.
    """

    return distance_backbone(D, weight=weight, algorithm="stlp", geometry="metric", distortion=distortion, self_loops=self_loops, cutoff=cutoff, verbose=verbose)


def ultrametric_backbone(D: nx.Graph | nx.DiGraph, weight: str = "weight", distortion: bool = False, self_loops: bool = False, cutoff: int = None, verbose: bool = False) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
    """
    Compute the ultrametric backbone of either a directed or an undirected weighted graph.

    This is a wrapper for :func:`distance_backbone`
    where ``kind="ultrametric"`` and ``algorithm="stlp"``.
    """

    return distance_backbone(D, weight=weight, algorithm="stlp", geometry="ultrametric", distortion=distortion, self_loops=self_loops, cutoff=cutoff, verbose=verbose)


# Private 
def _from_ttlp(D: nx.Graph | nx.DiGraph, weight: str, td_norm: Callable, distortion: bool, self_loops: bool, cutoff: int, verbose: bool) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
    """
    Compute the distance backbone using the Triangularity Tagging with Local Pruning (TTLP) algorithm.

    For each node in a weighted graph, use Dijkstra's algorithm to find the shortest path to each neighbor. 
    If the direct edge is not a shortest path, remove it. Otherwise, add it to an auxiliary graph.
    Stop when the auxiliary and input graphs have the same number of edges.

    Parameters
    ----------
    D : Directed or undirected NetworkX graph
        A directed or undirected weighted distance graph.
    weight : str
        Edge attribute containing distance values.
    td_norm : Callable
        Function used to measure path distance.
    distortion : bool
        Whether to compute edge distortion for edges not in the backbone.
    self_loops : bool
        Whether to remove semi-triangular self-loops.
    cutoff : int
        Restrict the maximum number of edges a shortest path can contain.
    verbose : bool
        Whether to display backbone computation progress.

    Returns
    -------
    nx.Graph or nx.DiGraph
        The distance backbone of either a directed or an undirected weighted graph.
    tuple of (nx.Graph or nx.DiGraph, dict)
        Distortions of edges not in the backbone, returned with the backbone when ``distortion=True``.
    """
        
    G = D.copy()
    B = nx.DiGraph() if nx.is_directed(G) else nx.Graph()

    if verbose: 
        total_nodes = G.number_of_nodes() 
        total_nodes_checked = 0

    for node in list(G.nodes()):
        shortest_distances = single_source_neighbors_dijkstra_path_length(G, source=node, weight=weight, td_norm=td_norm, cutoff=cutoff)
        
        for neighbor in list(G.neighbors(node)):
            direct_edge = G[node][neighbor]

            if shortest_distances[neighbor] < direct_edge[weight]:
                G.remove_edge(node, neighbor)
            else:
                B.add_edge(node, neighbor)

        if B.number_of_edges() == G.number_of_edges():
            break    

        if verbose:
            total_nodes_checked += 1
            total_progress = total_nodes_checked / total_nodes

            print(f"TTLP Backbone : dijkstra : {td_norm.__name__} : {total_nodes_checked} of {total_nodes} ({total_progress:.2%})")

    if self_loops:
        G = _remove_semi_triangular_self_loops(G, weight=weight, td_norm=td_norm)
    
    if distortion:
        distortions = _compute_distortions(D, G, weight=weight, td_norm=td_norm, self_loops=self_loops)
        return G, distortions

    return G


def _from_stlp(D: nx.Graph | nx.DiGraph, weight: str, td_norm: Callable, distortion: bool, self_loops: bool , cutoff: int, verbose: bool) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
    """
    Compute the distance backbone Semi-Triangular Local Pruning (STLP) algorithm.

    For each node in a weighted graph, use Dijkstra's algorithm to find the shortest path to each neighbor. 
    If the direct edge is not the shortest path, remove it.

    Parameters
    ----------
    D : Directed or undirected NetworkX graph
        A directed or undirected weighted distance graph.
    weight : str
        Edge attribute containing distance values.
    td_norm : Callable
        Function used to measure path distance.
    distortion : bool
        Whether to compute edge distortion for edges not in the backbone.
    self_loops : bool
        Whether to remove semi-triangular self-loops.
    cutoff : int
        Restrict the maximum number of edges a shortest path can contain.
    verbose : bool
        Whether to display backbone computation progress.

    Returns
    -------
    nx.Graph or nx.DiGraph
        The distance backbone of either a directed or an undirected weighted graph.
    tuple of (nx.Graph or nx.DiGraph, dict)
        Distortions of edges not in the backbone, returned with the backbone when ``distortion=True``.
    """
    
    G = D.copy()
    
    if verbose:
        total_nodes = G.number_of_nodes()
        total_nodes_checked = 0
    
    for node in list(G.nodes()):
        shortest_distances = single_source_neighbors_dijkstra_path_length(G, source=node, weight=weight, td_norm=td_norm, cutoff=cutoff)

        for neighbor in list(G.neighbors(node)):
            direct_edge = G[node][neighbor]

            if shortest_distances[neighbor] < direct_edge[weight]:
                G.remove_edge(node, neighbor)

        if verbose:
            total_nodes_checked += 1
            total_progress = total_nodes_checked / total_nodes

            print(f"STLP Backbone : dijkstra : {td_norm.__name__} : {total_nodes_checked} of {total_nodes} ({total_progress:.2%})")

    if self_loops:
        G = _remove_semi_triangular_self_loops(G, weight=weight, td_norm=td_norm)
     
    if distortion:
        distortions = _compute_distortions(D, G, weight=weight, td_norm=td_norm, self_loops=self_loops)    
        return G, distortions

    return G


def _from_closure(D: nx.Graph | nx.DiGraph, weight: str, geometry: str, td_norm: Callable, distortion: bool, self_loops: bool, cutoff: int, verbose: bool) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
    """
    Compute the distance backbone using the closure algorithm.

    Compute the distance closure. Remove direct edges that are not labeled as shortest paths.

    Parameters
    ----------
    D : Directed or undirected NetworkX graph
        A directed or undirected weighted distance graph.
    weight : str
        Edge attribute containing distance values.
    td_norm : Callable
        Function used to measure path distance.
    distortion : bool
        Whether to compute edge distortion for edges not in the backbone.
    self_loops : bool
        Whether to remove semi-triangular self-loops.
    cutoff : int
        Restrict the maximum number of edges a shortest path can contain.
    verbose : bool
        Whether to display backbone computation progress.

    Returns
    -------
    nx.Graph or nx.DiGraph
        The distance backbone of either a directed or an undirected weighted graph.
    tuple of (nx.Graph or nx.DiGraph, dict)
        Distortions of edges not in the backbone, returned with the backbone when ``distortion=True``.
    """
    
    G = D.copy()
    DC = distance_closure(G, geometry=geometry, weight=weight, existing_edges_only=True, self_loops=self_loops, verbose=verbose)

    internal_label = f"is_{geometry}"
    triangular_edges = [(source, target) for source, target in DC.edges() if DC[source][target][internal_label]]
    G = DC.edge_subgraph(triangular_edges).copy()
    
    if distortion:
        distortions = _compute_distortions(D, G, weight=weight, td_norm=td_norm, self_loops=self_loops)
        return G, distortions 

    return G


def _from_ksl(D: nx.Graph | nx.DiGraph, weight: str, td_norm: Callable, distortion: bool, self_loops: bool, cutoff: int, verbose: bool) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
    """
    Backbone computation combining triangle search based on "Kalavri-Simas-Logothetis (KSL), 2016, Proceedings of the VLDB Endowment, Volume 9, Issue 9".

    Parameters
    ----------
    D : Directed or undirected NetworkX graph
        A directed or undirected weighted distance graph.
    weight : str
        Edge attribute containing distance values.
    td_norm : Callable
        Function used to measure path distance.
    distortion : bool
        Whether to compute edge distortion for edges not in the backbone.
    self_loops : bool
        Whether to remove semi-triangular self-loops.
    cutoff : int
        Restrict the maximum number of edges a shortest path can contain.
    verbose : bool
        Whether to display backbone computation progress.

    Returns
    -------
    nx.Graph or nx.DiGraph
        The distance backbone of either a directed or an undirected weighted graph.
    tuple of (nx.Graph or nx.DiGraph, dict)
        Distortions of edges not in the backbone, returned with the backbone when ``distortion=True``.
    """

    G = D.copy()

    # Algorithm 1, page 676
    G = _remove_local_semi_triangular_edges(G, td_norm=td_norm, weight=weight)

    # Algorithm 2, page 677
    backbone_edges = _identify_local_triangular_edges(G, td_norm=td_norm, weight=weight)

    backbone = {(source, target) for source, target, _ in backbone_edges}
    unlabeled_edges = [(source, target) for source, target in G.edges() if (source, target) not in backbone]


    # Algorithm 3, page 677
    remaining_triangular_edges = []
    for source, target in unlabeled_edges:
        
        path = single_source_target_dijkstra_path(G, source=source, target=target, weight=weight, td_norm=td_norm, cutoff=cutoff)
        
        path_weights = [G[path[idx-1]][path[idx]][weight] for idx in range(1, len(path))]
        shortest_path_length = td_norm(path_weights)

        if G[source][target][weight] <= shortest_path_length:
            remaining_triangular_edges.append((source, target))
    
    final_edges = list(backbone) + remaining_triangular_edges
    G = G.edge_subgraph(final_edges).copy()

    if self_loops:
        G = _remove_semi_triangular_self_loops(G, weight=weight, td_norm=td_norm)

    if distortion:
        svals = _compute_distortions(D, G, weight=weight, td_norm=td_norm, self_loops=self_loops)
        return G, svals
    
    return G


def _from_approx(D: nx.Graph | nx.DiGraph, weight: str, td_norm: Callable, distortion: bool, self_loops: bool, cutoff: int) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
    """
    Backbone computation implementing a heuristic algorithm based on "Kalavri-Simas-Logothetis (KSL), 2016, Proceedings of the VLDB Endowment, Volume 9, Issue 9". 

    Parameters
    ----------
    D : Directed or undirected NetworkX graph
        A directed or undirected weighted distance graph.
    weight : str
        Edge attribute containing distance values.
    td_norm : Callable
        Function used to measure path distance.
    distortion : bool
        Whether to compute edge distortion for edges not in the backbone.
    self_loops : bool
        Whether to remove semi-triangular self-loops.
    cutoff : int
        Restrict the maximum number of edges a shortest path can contain.
    verbose : bool
        Whether to display backbone computation progress.

    Returns
    -------
    nx.Graph or nx.DiGraph
        The distance backbone of either a directed or an undirected weighted graph.
    tuple of (nx.Graph or nx.DiGraph, dict)
        Distortions of edges not in the backbone, returned with the backbone when ``distortion=True``.
    """
    
    G = D.copy()

    # Algorithm 1, page 676
    G = _remove_local_semi_triangular_edges(G, td_norm=td_norm, weight=weight)

    if self_loops:
        G = _remove_semi_triangular_self_loops(G, weight=weight, td_norm=td_norm)

    if distortion:
        distortions = _compute_distortions(D, G, weight=weight, td_norm=td_norm, self_loops=self_loops)
        return G, distortions 
    
    return G


def _remove_semi_triangular_self_loops(D: nx.Graph | nx.DiGraph, td_norm: Callable, weight: str) -> nx.Graph | nx.DiGraph:
    """
    Remove semi-triangular self-loops.

    Parameters
    ----------
    D : Directed or undirected NetworkX graph
        A directed or undirected weighted distance graph.
    td_norm : Callable
        Function used to measure path distance.
    weight : str
        Edge attribute containing distance values.
    
    Returns
    -------
    nx.Graph or nx.DiGraph
        Either a directed or an undirected weighted graph with semi-triangular self-loops removed.
    """

    edges_to_remove = []
    for node in nx.nodes_with_selfloops(D):
        self_loop = D[node][node]
        
        for neighbor in D.neighbors(node):
            if neighbor != node:
                path_length = single_source_neighbors_dijkstra_path_length(D, source=neighbor, weight=weight, td_norm=td_norm)
                direct_path = D[node][neighbor]
                alternative_loop_length = td_norm([ direct_path[weight], path_length[node] ])
                
                if alternative_loop_length < self_loop[weight]:
                    edges_to_remove.append((node, node))
                    break
                
    D.remove_edges_from(edges_to_remove)
    
    return D

def _remove_local_semi_triangular_edges(D: nx.Graph | nx.DiGraph, td_norm: Callable, weight: str) -> nx.Graph | nx.DiGraph:
    """
    Remove local semi-triangular edges. Algorithm 1 "Kalavri-Simas-Logothetis (KSL), 2016, Proceedings of the VLDB Endowment, Volume 9, Issue 9".

    Parameters
    ----------
    D : Directed or undirected NetworkX graph
        A directed or undirected weighted distance graph.
    td_norm : Callable
        Function used to measure path distance.
    weight : str
        Edge attribute containing distance values.

    Returns
    -------
    nx.Graph or nx.DiGraph
        Either a directed or an undirected weighted graph with local semi-triangular edges removed.
    """
    
    for node in D.nodes():
        neighbors = list(D.neighbors(node))
        triangles_to_check = product(neighbors, neighbors)
        
        for neighbor_1, neighbor_2 in triangles_to_check:
            if D.has_edge(node, neighbor_1) and D.has_edge(neighbor_2, neighbor_1) and D.has_edge(node, neighbor_2):
                direct_edge_1 = D[node][neighbor_1][weight]
                direct_edge_2 = D[node][neighbor_2][weight]
                connecting_edge = D[neighbor_2][neighbor_1][weight]

                if td_norm([ connecting_edge, direct_edge_2 ]) < direct_edge_1:
                    D.remove_edge(node, neighbor_1)
    return D


def _identify_local_triangular_edges(D: nx.Graph | nx.DiGraph, td_norm: Callable, weight: str) -> nx.Graph | nx.DiGraph:
    """
    Identify local triangular edges. Algorithm 2 of "Kalavri-Simas-Logothetis (KSL), 2016, Proceedings of the VLDB Endowment, Volume 9, Issue 9" 

    Parameters
    ----------
    D : Directed or undirected NetworkX graph
        A directed or undirected weighted distance graph.
    td_norm : Callable
        Function used to measure path distance.
    weight : str
        Edge attribute containing distance values.

    Returns
    -------
    nx.Graph or nx.DiGraph
        Either a directed or an undirected weighted graph with local semi-triangular edges removed.
    """
    
    U = {}
    for node in D.nodes():
        neighbors = [(node, target, data[weight]) for target, data in D[node].items()]
        U[node] = sorted(neighbors, key=lambda item: item[2])

    triangular_edges = set()
    for node in D.nodes():
        if not U[node]:
            continue

        weights_for_comparison = set()
        triangular = True

        removed_pair = U[node].pop(0)
        triangular_edges.add(removed_pair)
        
        while U[node]:
            e = U[node].pop(0)
            for _, target, _ in triangular_edges:
                if D.has_edge(node, target) and U[target]:
                    w_x = td_norm([ D[node][target][weight], U[target][0][2] ])
                    weights_for_comparison.add(w_x)
            
            for w in weights_for_comparison:
                if e[2] > w:
                    triangular = False
                    break

            if triangular:
                triangular_edges.add(e)
                weights_for_comparison = set()
            else:
                return triangular_edges
            
    return triangular_edges


def _compute_distortions(D: nx.Graph | nx.DiGraph, B: nx.Graph | nx.DiGraph, td_norm: Callable, weight: str, self_loops: bool) -> dict[tuple[Hashable, Hashable], float]:
    """
    Compute distortions of edges not in backbone.

    Parameters
    ----------
    D : Directed or undirected NetworkX graph
        A directed or undirected weighted distance graph.
    B : Directed or undirected NetworkX graph
        A directed or undirected weighted distance graph.
    td_norm: Callable
        Function used to measure path distance.
    weight : str
        Edge attribute containing distance values.
    self_loops : bool
        Whether to remove semi-triangular self-loops.

    Returns
    -------
    dict
        Distortions of edges not in the backbone.
    
    """
    G = D.copy()
    G.remove_edges_from(B.edges())

    all_self_loops = dict()
    if self_loops:
        for node in nx.nodes_with_selfloops(G):
            self_loop = G[node][node]
            
            for neighbor in G.neighbors(node):
                return_path_length = single_source_neighbors_dijkstra_path_length(B, source=neighbor, weight=weight, td_norm=td_norm, cutoff=None)
                direct_path = G[node][neighbor]
                alternative_self_loop = td_norm([ direct_path[weight], return_path_length[node] ])
                
                if alternative_self_loop < self_loop[weight]:
                    self_loop[weight] = alternative_self_loop
            
            all_self_loops[node] = self_loop

    distortions = dict()        
    for node in G.nodes():
        shortest_distances = single_source_neighbors_dijkstra_path_length(B, source=node, weight=weight, td_norm=td_norm, cutoff=None)
        for neighbor in G.neighbors(node):
            if (neighbor == node) and self_loops:
                distortions[(node, neighbor)] = G[node][neighbor][weight] / all_self_loops[node]
            else:
                distortions[(node, neighbor)] = G[node][neighbor][weight] / shortest_distances[neighbor]
    
    return distortions   


_BACKBONE_ALGORITHMS = {
    "stlp": _from_stlp,
    "ttlp": _from_ttlp,
    "closure": _from_closure,
    "ksl": _from_ksl,
    "approx": _from_approx
}

_GEOMETRY = {
    "metric": sum,
    "ultrametric": max,
}
