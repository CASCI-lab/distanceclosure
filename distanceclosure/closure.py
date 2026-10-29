# -*- coding: utf-8 -*-
"""
Distance Closure
================

Compute the distance closure of a weighted graph.
"""

import numpy as np
import networkx as nx
from typing import Callable
from distanceclosure.dijkstra import all_pairs_dijkstra_path_length, single_source_target_dijkstra_path


__all__ = [
    "distance_closure"
]

def distance_closure(D: nx.Graph | nx.DiGraph, weight='weight', geometry='metric', existing_edges_only=False, self_loops=False, verbose=False) -> nx.Graph | nx.DiGraph: 
    """
    Compute the distance closure of a weighted graph.

    Parameters
    ----------
    D : nx.Graph or nx.DiGraph
        Directed or undirected distance graph.
    weight : str, optional
        Edge attribute containing distance values. Default is ``"weight"``.
    geometry : {"metric", "ultrametric"}, optional
        Type of distance closure to compute. Default is ``"metric"``.
    existing_edges_only : bool, optional
        If False, add closure edges between reachable non-adjacent nodes.
    self_loops : bool, optional
        If True, compute the distance closure considering self-loops.
    verbose : bool, optional
        If True, print progress information during computation.
        Default is ``False``.

    Returns
    -------
    nx.Graph or nx.DiGraph
        Copy of the input graph with additional two additional edge attributes.
    
    Raises
    ------
    ValueError
        If ``kind`` is invalid.
    """

    try:
        chosen_td_norm = _GEOMETRY[geometry]
    except KeyError:
        raise ValueError(f"Invalid input. Valid arguments are: {list(_GEOMETRY)}")

    return _closure(D, geometry=geometry, td_norm=chosen_td_norm, weight=weight, existing_edges_only=existing_edges_only, self_loops=self_loops, verbose=verbose)


def _closure(D: nx.Graph | nx.DiGraph, geometry: str, td_norm: Callable, weight: str, existing_edges_only: bool, self_loops: bool, verbose: bool) -> nx.Graph | nx.DiGraph:
    """
    Compute the distance closure using all-pairs shortest paths (APSP)
    with two different shortest path measures on a weighted distance graph.

    Parameters
    ----------
    D : nx.Graph or nx.DiGraph
        Directed or undirected distance graph.
    kind : {"metric", "ultrametric"}
        Type of distance closure to compute.
    disjunction : {sum, max}
        Function used to measure distance. ``sum`` 
        for metric and ``max`` for ultrametric.
    weight : str
        Edge attribute containing distance values.
    existing_edges_only : bool, optional
        If False, add closure edges between reachable non-adjacent nodes.
    self_loops : bool, optional
        If True, compute the distance closure considering self-loops.
    verbose : bool, optional
        If True, print progress information during computation.

    Returns
    -------
    nx.Graph or nx.DiGraph
        Copy of the input graph with two additional edge attributes:
        ``metric_distance`` and ``is_metric``, or ``ultrametric_distance``
        and ``is_ultrametric``, depending on the value of ``kind``.

        Edges added due to ``existing_edges_only=False`` receive only the ``metric_distance`` 
        or ``ultrametric_distance`` attribute.
        
    """
    if verbose:
        total_nodes = G.number_of_nodes()
        total_nodes_checked = 1

    G = D.copy() 
    edges_seen = set()
    edges_to_add = []

    for u, lengths in all_pairs_dijkstra_path_length(G, weight=weight, disjunction=td_norm):
        if verbose:
            total_progress = total_nodes_checked / total_nodes
            print(f"Distance Closure : dijkstra : {geometry} : {total_nodes_checked} of {total_nodes} ({total_progress:.2%})")
            
        for v, length in lengths.items():
            if (u, v) in edges_seen or u == v:
                continue
            else:
                edges_seen.add((u, v))
                distance_label = f"{geometry}_distance"
                geometry_label = f"is_{geometry}"

                if not G.has_edge(u, v):
                    if not existing_edges_only:
                        edges_to_add.append((u, v, {weight: np.inf, distance_label: length}))
                else:
                    G[u][v][distance_label] = length
                    G[u][v][geometry_label] = True if (length == G[u][v][weight]) else False
            total_nodes_checked += 1

    if self_loops:
        for u in G.nodes():
            if not G.has_edge(u, u):
                length = np.inf
            else:
                length = G[u][u][weight]

            for k in G.neighbors(u):
                return_path = single_source_target_dijkstra_path(G, source=k, target=u, weight=weight, disjunction=td_norm)
                spl = td_norm([G[u][k][weight], td_norm(G[return_path[idx-1]][return_path[idx]][weight] for idx in range(1, len(return_path)))])
                if spl < length:
                    length = spl
            
            distance_label = '{kind:s}_distance'.format(kind=geometry)
            geometry_label = 'is_{kind:s}'.format(kind=geometry)
            if not G.has_edge(u, u):
                if not existing_edges_only:
                    edges_to_add.append((u, u, {weight: np.inf, distance_label: length}))
            else:
                G[u][u][distance_label] = length
                G[u][u][geometry_label] = True if (length == G[u][u][weight]) else False
    
    G.add_edges_from(edges_to_add)

    return G


_GEOMETRY = {
    "metric": sum,
    "ultrametric": max,
}