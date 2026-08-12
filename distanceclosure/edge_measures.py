# -*- coding: utf-8 -*-
"""
Transitive Closure
==================

Computes transitive closure on a weighted graph.
These algorithms work with undirected weighted (distance) graphs.
"""

import numpy as np
import networkx as nx
from distanceclosure.dijkstra import all_pairs_dijkstra_path_length
__name__ = 'distanceclosure'
__author__ = """\n""".join(['Rion Brattig Correia <rionbr@gmail.com>'])

__all__ = [
    "s_values",
    "b_values"
]


__kinds__ = ['metric', 'ultrametric']
__algorithms__ = ['dense', 'dijkstra']


def _check_for_kind(kind):
    """
    Check for available metric functions.
    """
    if kind not in __kinds__:
        raise TypeError("Metric not found for this algorithm. Try 'metric' or 'ultrametric',")


def _check_for_algorithm(algorithm):
    """
    Check for available algorithm.
    """
    if algorithm not in __algorithms__:
        raise TypeError("Algorithm implementation not supported. Try 'dense', 'dijkstra' or leave blank.")


def s_values(Cm, weight_distance='distance', weight_metric_distance='metric_distance'):
    """
    Computes s-values for each network edge.
    The s-value is the ratio between the direct distance (from the original graph) and the indirect distance (from the metric distance closure graph).
    The formal definition is as follow:

    .. math::
        s_{ij} = d_{ij} / d_{ij}^m.

    Args:
        Cm (networkx.Graph): The metric distance closure graph.
        weight_distance (string): Edge attribute containing distance values. Defaults to 'distance'.
        weight_metric_distance (string): Edge attribute containing metric distance values. Defaults to 'metric_distance'.
    """
    G = Cm.copy()
    #
    dict_s_values = {
        (i, j): d.get(weight_distance) / d.get(weight_metric_distance)
        for i, j, d in G.edges(data=True)
        if ((d.get(weight_distance) < np.inf) and (d.get(weight_metric_distance) > 0))
    }
    nx.set_edge_attributes(G, name='s-value', values=dict_s_values)

    return G


def b_values(Cm, weight_distance='distance', weight_metric_distance='metric_distance'):
    """Computes b-values for each edge with infinite distance, thus not existing in the original distance graph.
    The formal definition is as follow:

    .. math::
        b_{ij} = <d_{ik}> / d_{ij}^m

        b_{ji} = <d_{jk}> / d_{ij}^m

    which is the average distance of all edges that leaves from node `x_i`, divided by its metric distance closure.
    Note that `b_{ij}` can be different from `b_{ji}`.

    Parameters
    ----------
    Cm (networkx.Graph): The metric distance closure graph.
    weight_distance (string): Edge attribute containing distance values. Defaults to 'distance'.
    weight_metric_distance (string): Edge attribute containing metric distance values. Defaults to 'metric_distance'.

    Note
    ----
    Both arguments must be numpy arrays as the Metric Closure network is a dense matrix.

    Warning
    -------
    This computation takes a while.
    """
    G = Cm.copy()

    mean_distance = {
        k: np.mean([d.get(weight_distance) for i, j, d in G.edges(nbunch=k, data=True) if d.get(weight_distance, None) < np.inf])
        for k in G.nodes()
    }

    dict_b_ij_values = {
        (i, j): mean_distance[i] / d.get(weight_metric_distance)
        for i, j, d in G.edges(data=True)
        if (d.get(weight_distance) == np.inf)
    }
    nx.set_edge_attributes(G, name='b_ij-value', values=dict_b_ij_values)

    dict_b_ji_values = {
        (i, j): mean_distance[j] / d.get(weight_metric_distance)
        for i, j, d in G.edges(data=True)
        if (d.get(weight_distance) == np.inf)
    }
    nx.set_edge_attributes(G, name='b_ji-value', values=dict_b_ji_values)

    return G
