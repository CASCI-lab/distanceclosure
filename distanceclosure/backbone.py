# -*- coding: utf-8 -*-
"""
Backbone Subgraph - Fast Implementation
==================

Computes the shortest path distance backbone on a weighted graph.
These algorithms work with edges weighted as distances.
"""

import numpy as np
import networkx as nx
from distanceclosure.dijkstra import single_source_dijkstra_path_length
from networkx.algorithms.shortest_paths.weighted import _weight_function

__name__ = 'distanceclosure'
__author__ = """\n""".join(['Rion Brattig Correia <rionbr@gmail.com>', 'Felipe Xavier Costa <fcosta@binghamton.com>'])

__all__ = [
    "metric_backbone",
    "ultrametric_backbone"
]


__kinds__ = ['metric', 'ultrametric']
__algorithms__ = ['dense', 'dijkstra']


def metric_backbone(D, weight='weight', self_loops=False, cutoff=None, verbose=False, *args, **kwargs):
    """
    Fast backbone (only) computation considering node ordering.

    Parameters
    ----------
    D : NetworkX graph
        The Distance graph
    weight : str, optional
        Edge property containing distance values, by default 'weight'
    self_loops : bool, optional
        If the distance graph has nodes with self distance greater than zero, by default False
    cutoff : _type_, optional
        Maximum number of connections in the path. If None, compute the entire closure as is the cutoff is the number of nodes, by default None
    verbose : bool, optional
        Prints statements as it computes, by default False

    Returns
    -------
    NetworkX graph
        The metric backbone subgraph.

    Raises
    ------
    NotImplementedError
        Self-loop closure and finite step (cutoff) not implemented yet
    """

    if self_loops:
        raise NotImplementedError
    if cutoff is not None:
        raise NotImplementedError

    G = _compute_backbone(D, weight=weight, disjunction=sum, verbose=False, *args, **kwargs)

    return G

def ultrametric_backbone(D, weight='weight', self_loops=False, cutoff=None, verbose=False, *args, **kwargs):
    """
    Fast backbone (only) computation considering node ordering.

    Parameters
    ----------
    D : NetworkX graph
        The Distance graph
    weight : str, optional
        Edge property containing distance values, by default 'weight'
    self_loops : bool, optional
        If the distance graph has nodes with self distance greater than zero, by default False
    cutoff : _type_, optional
        Maximum number of connections in the path. If None, compute the entire closure as is the cutoff is the number of nodes, by default None
    verbose : bool, optional
        Prints statements as it computes, by default False

    Returns
    -------
    NetworkX graph
        The ultrametric backbone subgraph.

    Raises
    ------
    NotImplementedError
        Self-loop closure and finite step (cutoff) not implemented yet
    """

    if self_loops:
        raise NotImplementedError
    if cutoff is not None:
        raise NotImplementedError

    G = _compute_backbone(D, weight=weight, disjunction=max, verbose=False, *args, **kwargs)

    return G

def _compute_backbone(D, weight='weight', disjunction=sum, self_loops=False, cutoff=None, verbose=False, *args, **kwargs):
    """
    Fast backbone (only) computation considering node ordering.

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
    
    G = D.copy()

    ordered_nodes = sorted(G.degree(weight=weight), key=lambda x: x[1], reverse=True)

    weight_function = _weight_function(G, weight)
    
    if verbose:
        total = G.number_of_nodes()
        i = 0

    for n, _ in ordered_nodes:
        if verbose:
            i += 1
            per = i/total
            print("Closure: Dijkstra : source node {u:s} : {i:d} of {total:d} ({per:.2%})".format(u=n, i=i, total=total, per=per))

        metric_dist = single_source_dijkstra_path_length(G, source=n, weight_function=weight_function, disjunction=disjunction, cutoff=cutoff)
        neighbors = list(G.neighbors(n)) # Need to be separate or will raise changing list error
        for v in neighbors:
            if metric_dist[v] < G[n][v][weight]:
                G.remove_edge(n, v)
    
    return G