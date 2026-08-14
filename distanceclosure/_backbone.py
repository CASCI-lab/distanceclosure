"""
Algorithms
============================

Algorithms and their helper functions
"""

import numpy as np
import networkx as nx
from distanceclosure._dijkstra import _single_source_dijkstra_path_length, _single_source_target_dijkstra_path, _single_source_neighbors_dijkstra_path_length, _all_pairs_dijkstra_path_length
from distanceclosure.closure import distance_closure
from itertools import product
from typing import Callable

# Algorithms

def _flagged_backbone(D: nx.Graph | nx.DiGraph, weight: str, disjunction: Callable, distortion: bool, self_loops: bool, cutoff: int, verbose: bool, *args, **kwargs) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
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

    
def _iterative_backbone(D: nx.Graph | nx.DiGraph, weight: str, disjunction: Callable, distortion: bool, self_loops: bool , cutoff: int, verbose: bool, *args, **kwargs) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
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


def _closure_backbone(D: nx.Graph | nx.DiGraph, weight: str, disjunction: Callable, distortion: bool, self_loops: bool, cutoff: int, verbose: bool, *args, **kwargs) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
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

    DC = distance_closure(D, kind=kind, algorithm='dijkstra', weight=weight, only_backbone=True, verbose=verbose, *args, **kwargs)

    is_kind = 'is_{kind:s}'.format(kind=kind)
    metric_edges = [(u, v) for u, v in DC.edges() if DC[u][v][is_kind]]
    G = DC.edge_subgraph(metric_edges).copy()
    
    if distortion:
        svals = _compute_distortions(D, G, weight=weight, kind=kind)         
        return G, svals

    return G


def _heuristic_backbone(D: nx.Graph | nx.DiGraph, weight: str, disjunction: Callable, distortion: bool, self_loops: bool, cutoff: int, verbose: bool, *args, **kwargs) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
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


def _approximate_backbone(D: nx.Graph | nx.DiGraph, weight: str, disjunction: Callable, distortion: bool, self_loops: bool, cutoff: int, *args, **kwargs) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
    G = D.copy()

    # Algorithm 1, page 676
    G = _local_semi_triangles(G, disjunction=disjunction, weight=weight)

    # Compute Distortion
    if distortion:
        svals = _compute_distortions(D, G, weight=weight, disjunction=disjunction)
        return G, svals
    
    return G


# Helper Functions

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


