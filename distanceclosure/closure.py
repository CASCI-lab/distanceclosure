# -*- coding: utf-8 -*-
"""
Distance Closure
================

Compute the distance closure of a weighted graph.

Authors
-------
Rion Brattig Correia <rionbr@gmail.com>
Felipe Xavier Costa <fcosta@binghamton.edu>
Robert Palermo <rpalermo@binghamton.edu>
"""

import numpy as np
import networkx as nx
from typing import Callable
from distanceclosure.dijkstra import all_pairs_dijkstra_path_length, single_source_target_dijkstra_path


__all__ = [
    "distance_closure"
]


def distance_closure(D: nx.Graph | nx.DiGraph, weight: str = "weight", geometry: str = "metric", existing_edges_only: bool = True, self_loops: bool = False, cutoff: int = None, verbose: bool = False) -> nx.Graph | nx.DiGraph: 
    """
    Compute the distance closure of either a directed or an undirected weighted graph.

    Parameters
    ----------
    D : Directed or undirected NetworkX graph
        A directed or undirected weighted distance graph.
    weight : str, optional
        Edge attribute containing distance values. ``weight="weight"`` by default.
    geometry : {"metric", "ultrametric"}, optional
        Distance metric used to compute the closure. "metric" uses sum and "ultrametric" uses max. ``kind="metric"`` by default.
    existing_edges_only : bool, optional
        Whether to add closure edges between reachable non-adjacent nodes. ``existing_edges_only=True`` by default.
    self_loops : bool, optional
        Whether to remove semi-triangular self-loops.``self_loops=False`` by default.
    verbose : bool, optional
        Whether to display closure computation progress. ``verbose=False`` by default.

    Returns
    -------
    nx.Graph or nx.DiGraph
        Copy of the input directed or undirected weighted graph with two additional edge attributes:
        ``metric_distance`` and ``is_metric``, or ``ultrametric_distance``
        and ``is_ultrametric``, depending on the value of ``geometry``.

        Edges added due to ``existing_edges_only=False`` receive only the ``metric_distance`` 
        or ``ultrametric_distance`` attribute.  
    
    Raises
    ------
    ValueError
        If ``geometry`` is not a valid key in ``_GEOMETRY``.
    """

    try:
        chosen_td_norm = _GEOMETRY[geometry]
    except KeyError:
        raise ValueError(f"Invalid input. Valid arguments are: {list(_GEOMETRY)}")

    return _closure(D, geometry=geometry, td_norm=chosen_td_norm, weight=weight, existing_edges_only=existing_edges_only, self_loops=self_loops, cutoff=cutoff, verbose=verbose)


def _closure(D: nx.Graph | nx.DiGraph, weight: str, geometry: str, td_norm: Callable, existing_edges_only: bool, self_loops: bool, cutoff: int, verbose: bool) -> nx.Graph | nx.DiGraph:
    """
    Compute the distance closure using all-pairs shortest paths (APSP).

    Parameters
    ----------
    D : Directed or undirected NetworkX graph
        A directed or undirected weighted distance graph.
    weight : str
        Edge attribute containing distance values.
    td_norm : Callable
        Function used to measure path distance.
    distortion : bool
        Whether to compute edge distortion for edges not in the closure.
    self_loops : bool
        Whether to remove semi-triangular self-loops.
    cutoff : int
        Restrict the maximum number of edges a shortest path can contain.
    verbose : bool
        Whether to display closure computation progress.

    Returns
    -------
    nx.Graph or nx.DiGraph
        Copy of the input directed or undirected weighted graph with two additional edge attributes:
        ``metric_distance`` and ``is_metric``, or ``ultrametric_distance``
        and ``is_ultrametric``, depending on the value of ``geometry``.

        Edges added due to ``existing_edges_only=False`` receive only the ``metric_distance`` 
        or ``ultrametric_distance`` attribute.      
    """
    
    if verbose:
        total_nodes = G.number_of_nodes()
        total_nodes_checked = 1

    G = D.copy() 
    edges_seen = set()
    edges_to_add = []

    for u, lengths in all_pairs_dijkstra_path_length(G, weight=weight, td_norm=td_norm, cutoff=cutoff):
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
        
        if verbose:
            total_nodes_checked += 1
            total_progress = total_nodes_checked / total_nodes

            print(f"Distance Closure : dijkstra : {geometry} : {total_nodes_checked} of {total_nodes} ({total_progress:.2%})")

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