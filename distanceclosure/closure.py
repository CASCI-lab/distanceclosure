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

def distance_closure(D: nx.Graph | nx.DiGraph, kind='metric', weight='weight', existing_edges_only=False, self_loops=False, verbose=False) -> nx.Graph | nx.DiGraph: 
    """
    Compute the distance closure of a weighted graph.

    Parameters
    ----------
    D : nx.Graph or nx.DiGraph
        Directed or undirected distance graph.
    kind : {"metric", "ultrametric"}, optional
        Type of distance closure to compute. Default is ``"metric"``.
    weight : str, optional
        Edge attribute containing distance values. Default is ``"weight"``.
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
        disjunction = _KINDS[kind]
    except KeyError:
        raise ValueError("Invalid input. Valid arguments are: {_KINDS:s}".format(_KINDS=_KINDS.keys()))

    return _closure(D, kind=kind, disjunction=disjunction, weight=weight, existing_edges_only=existing_edges_only, self_loops=self_loops, verbose=verbose)


# Private
def _closure(D: nx.Graph | nx.DiGraph, kind: str, disjunction: Callable, weight: str, existing_edges_only: bool, self_loops: bool, verbose: bool) -> nx.Graph | nx.DiGraph:
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

    G = D.copy() 
    edges_seen = set()
    total = G.number_of_nodes()

    i = 1
    for u, lengths in all_pairs_dijkstra_path_length(G, weight=weight, disjunction=disjunction):
        if verbose:
            per = i / total
            print("Distance Closure : dijkstra : {kind:s} : {i:d} of {total:d} ({per:.2%})".format(kind=kind, i=i, total=total, per=per))
        for v, length in lengths.items():
            if (u, v) in edges_seen or u == v:
                continue
            else:
                edges_seen.add((u, v))
                kind_distance = '{kind:s}_distance'.format(kind=kind)
                is_kind = 'is_{kind:s}'.format(kind=kind)

                if not G.has_edge(u, v):
                    if not existing_edges_only:
                        G.add_edge(u, v, **{weight: np.inf, kind_distance: length})
                else:
                    G[u][v][kind_distance] = length
                    G[u][v][is_kind] = True if (length == G[u][v][weight]) else False
        i += 1

    if self_loops:
        for u in G.nodes():
            if not G.has_edge(u, u):
                length = np.inf
            else:
                length = G[u][u][weight]

            for k in G.neighbors(u):
                return_path = single_source_target_dijkstra_path(G, source=k, target=u, weight=weight, disjunction=disjunction)
                spl = disjunction([G[u][k][weight], disjunction(G[return_path[idx-1]][return_path[idx]][weight] for idx in range(1, len(return_path)))])
                if spl < length:
                    length = spl

            if not G.has_edge(u, u):
                if not existing_edges_only:
                    G.add_edge(u, u, **{weight: np.inf, kind_distance: length})
            else:
                G[u][u][kind_distance] = length
                G[u][u][is_kind] = True if (length == G[u][u][weight]) else False

    return G


_KINDS = {
    "metric": sum,
    "ultrametric": max,
}