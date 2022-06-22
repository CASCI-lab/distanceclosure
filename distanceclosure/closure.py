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
    "distance_closure",
    "s_values",
    "b_values"
]


__kinds__ = ['metric', 'ultrametric']
__algorithms__ = ['dense', 'dijkstra']


def distance_closure(D, kind='metric', algorithm='dijkstra', weight='weight', only_backbone=False, verbose=False, *args, **kwargs):
    """Computes the transitive closure (All-Pairs-Shortest-Paths; APSP)
    using different shortest path measures on the distance graph
    (adjacency matrix) with values in the ``[0,inf]`` interval.

    .. math::

        c_{ij} = min_{k}( metric ( a_{ik} , b_{kj} ) )

    Parameters
    ----------
    D : NetworkX.Graph
        The Distance graph.

    kind : string
        Type of closure to compute: ``metric`` or ``ultrametric``.

    algorithm : string
        Type of algorithm to use: ``dense`` or ``dijkstra``.

    weight : string
        Edge property containing distance values. Defaults to `weight`.
    
    only_backbone : bool
        Only include new distance closure values for edges in the original graph.
    
    Verbose :bool
        Prints statements as it computes.

    Returns
    --------
    C : NetworkX.Graph
        The distance closure graph. Note this may be a fully connected graph.

    Examples
    --------
    >>> distance_closure(D, kind='metric', algorithm='dijkstra', weight='weight', only_backbone=True)

    Note
    ----
    Dense matrix is slow for large graphs.
    We are currently working on optimizing it.
    If your network is large and/or sparse, use the Dijkstra method.

    - Metric: :math:`(min,+)`
    - Ultrametric: :math:`(min,max)` -- also known as maximum flow.
    - Semantic proximity: (to be implemented)

    .. math::

            [ 1 + \\sum_{i=2}^{n-1} log k(v_i) ]^{-1}
    """
    _check_for_kind(kind)
    _check_for_algorithm(algorithm)

    G = D.copy()

    # Dense
    if algorithm == 'dense':

        raise NotImplementedError('Needs some fine tunning.')
        #M = nx.to_numpy_matrix(D, *args, **kwargs)
        #return _transitive_closure_dense_numpy(M, kind, *args, **kwargs)

    # Dijkstra
    elif algorithm == 'dijkstra':

        if kind == 'metric':
            disjunction = sum
        elif kind == 'ultrametric':
            disjunction = max

        edges_seen = set()
        i = 1
        total = G.number_of_nodes()
        # APSP
        for u, lengths in all_pairs_dijkstra_path_length(G, weight=weight, disjunction=disjunction):
            if verbose:
                per = i / total
                print("Closure: Dijkstra : {kind:s} : source node {u:s} : {i:d} of {total:d} ({per:.2%})".format(kind=kind, u=u, i=i, total=total, per=per))
            for v, length in lengths.items():

                if (u, v) in edges_seen or u == v:
                    continue
                else:
                    edges_seen.add((u, v))
                    kind_distance = '{kind:s}_distance'.format(kind=kind)
                    is_kind = 'is_{kind:s}'.format(kind=kind)
                    if not G.has_edge(u, v):
                        if not only_backbone:
                            G.add_edge(u, v, **{weight: np.inf, kind_distance: length})
                    else:
                        G[u][v][kind_distance] = length
                        G[u][v][is_kind] = True if (length == G[u][v][weight]) else False
            i += 1

    return G


def _transitive_closure_dense_numpy(A, kind='metric', verbose=False):
    """
    Calculates Transitive Closure using numpy dense matrix traversing.
    """
    C = A.copy()
    n, m = A.shape

    # Check if diagonal is all zero
    if sum(np.diagonal(A)) > 0:
        raise ValueError("Diagonal has to be zero for matrix computation to be correct")

    # Compute Transitive Closure
    for i in range(0, n):
        if verbose:
            print('calc row:', i + 1, 'of', m)
        for j in range(0, n):

            if kind == 'metric':
                vec = C[i, :] + C[:, j]
                C[i, j] = vec.min()

            elif kind == 'ultrametric':
                vec = np.maximum(C[i, :], C[:, j])
                C[i, j] = vec.min()

    return np.array(C)


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
