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

def distance_closure(D: nx.Graph | nx.DiGraph, weight: str = "weight", td_name: str = "metric", existing_edges_only: bool = True, self_loops: bool = False, cutoff: int = None, verbose: bool = False) -> nx.Graph | nx.DiGraph: 
    """
    Compute the distance closure of either a directed or an undirected weighted graph.

    Parameters
    ----------
    D : Directed or undirected NetworkX graph
        A directed or undirected weighted distance graph.
    weight : str, optional
        Edge attribute containing distance values. ``weight="weight"`` by default.
    td_name : {"metric", "ultrametric"}, optional
        Distance metric used to compute the closure. "metric" uses sum and "ultrametric" uses max. ``td_name="metric"`` by default.
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
        and ``is_ultrametric``, depending on the value of ``td_name``.

        Edges added due to ``existing_edges_only=False`` receive only the ``metric_distance`` 
        or ``ultrametric_distance`` attribute.  
    
    Raises
    ------
    ValueError
        If ``td_name`` is not a valid key in ``_TD_NAME``.
    """

    try:
        chosen_td_norm = _TD_NAME[td_name]
    except KeyError:
        raise ValueError(f"Invalid input. Valid arguments are: {list(_TD_NAME)}")

    return _closure(D, td_name=td_name, td_norm=chosen_td_norm, weight=weight, existing_edges_only=existing_edges_only, self_loops=self_loops, cutoff=cutoff, verbose=verbose)


def _closure(D: nx.Graph | nx.DiGraph, weight: str, td_name: str, td_norm: Callable[[float, float], float], existing_edges_only: bool, self_loops: bool, cutoff: int, verbose: bool) -> nx.Graph | nx.DiGraph:
    """
    Compute the distance closure using all-pairs shortest paths (APSP).

    Parameters
    ----------
    D : Directed or undirected NetworkX graph
        A directed or undirected weighted distance graph.
    weight : str
        Edge attribute containing distance values.
    td_norm : Callable[[float, float], float]
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
        and ``is_ultrametric``, depending on the value of ``td_name``.

        Edges added due to ``existing_edges_only=False`` receive only the ``metric_distance`` 
        or ``ultrametric_distance`` attribute.      
    """

    G = D.copy() 
    edges_seen = set()
    edges_to_add = []

    if verbose:
        total_nodes = G.number_of_nodes()
        total_nodes_checked = 1
        
    for source, lengths in all_pairs_dijkstra_path_length(G, weight=weight, td_norm=td_norm, cutoff=cutoff):
        for target, length in lengths.items():
            if (source, target) in edges_seen or source == target:
                continue
            else:
                edges_seen.add((source, target))
                distance_label = f"{td_name}_distance"
                geometry_label = f"is_{td_name}"

                if not G.has_edge(source, target):
                    if not existing_edges_only:
                        edges_to_add.append((source, target, {weight: np.inf, distance_label: length}))
                else:
                    G[source][target][distance_label] = length
                    G[source][target][geometry_label] = True if (length == G[source][target][weight]) else False
        
        if verbose:
            total_nodes_checked += 1
            total_progress = total_nodes_checked / total_nodes

            print(f"Distance Closure : dijkstra : {td_name} : {total_nodes_checked} of {total_nodes} ({total_progress:.2%})")

    if self_loops:
        for source in G.nodes():
            if not G.has_edge(source, source):
                length = np.inf
            else:
                length = G[source][source][weight]

            for k in G.neighbors(source):
                if k == source or not nx.has_path(G, k, source):
                    continue
                
                return_path = single_source_target_dijkstra_path(G, source=k, target=source, weight=weight, td_norm=td_norm, cutoff=None)
                spl = td_norm([G[source][k][weight], td_norm(G[return_path[idx-1]][return_path[idx]][weight] for idx in range(1, len(return_path)))])
                if spl < length:
                    length = spl
            
            distance_label = '{kind:s}_distance'.format(kind=td_name)
            geometry_label = 'is_{kind:s}'.format(kind=td_name)
            if not G.has_edge(source, source):
                if not existing_edges_only:
                    edges_to_add.append((source, source, {weight: np.inf, distance_label: length}))
            else:
                G[source][source][distance_label] = length
                G[source][source][geometry_label] = True if (length == G[source][source][weight]) else False
    
    G.add_edges_from(edges_to_add)

    return G


_TD_NAME = {
    "metric": sum,
    "ultrametric": max,
}