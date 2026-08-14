# -*- coding: utf-8 -*-
"""
Utils
==========================

Utility functions for the Distance Closure package
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import networkx as nx

__all__ = [
    'prox2dist',
    'dist2prox',
    'dict2matrix',
    'matrix2dict',
    'dict2sparse',
    'from_networkx_to_dijkstra_format',
    's_values',
    'b_values'
]


def prox2dist(p):
    """Transforms a non-negative ``[0,1]`` proximity to distance in the ``[0,inf]`` interval:

    .. math::

        d = \\frac{1}{p} - 1

    Parameters
    ----------
    p : float
        Proximity value

    Returns
    -------
    d : float
        Distance value

    See Also
    --------
    dist2prox
    """
    if (p == 0):
        return np.inf
    else:
        return (1 / float(p)) - 1


def dist2prox(d):
    """
    Transforms a non-negative integer distance ``d`` to a proximity/similarity value in the ``[0,1]`` interval:

    .. math::

        p = \\frac{1}{(d+1)}

    It accepts both dense and sparse matrices.

    Parameters
    ----------
    D :matrix
        Distance matrix

    Returns
    -------
    P : matrix
        Proximity matrix

    See Also
    --------
    prox2dist

    """
    if d == np.inf:
        return 0
    else:
        return (d + 1) ** -1


def dict2matrix(d):
    """
    Tranforms a 2D dictionary into a numpy. Usefull when converting Dijkstra results.

    Parameters
    ----------
        d (dict): 2D dictionary

    Returns
    -------
    m : Numpy matrix

    Warning
    -------
    If your nodes are identified by names instead of numbers, make sure to keep a mapping.

    Examples
    --------
    >>> d = {0: {0: 0, 1: 1, 2:3}, 1: {0: 1, 1: 0, 2:2}, 2: {0: 3, 1:2, 2:0}}
    >>> dict2matrix(d)
        [[ 0 1 3]
         [ 1 0 2]
         [ 3 2 0]]

    Note
    ----
    Uses pandas to accomplish this in a one liner.

    See Also
    --------
    matrix2dict
    """
    return pd.DataFrame.from_dict(d).values


def matrix2dict(m):
    """
    Tranforms a Numpy matrix into a 2D dictionary. Usefull when comparing dense metric and Dijkstra results.

    Parameters
    ----------
        m (matrix): numpy matrix

    Returns
    -------
        d (dict): 2D dictionary

    Examples
    --------
    >>> m = [[0, 1, 3], [1, 0, 2], [3, 2, 0]]
    >>> matrix2dict(m)
        {0: {0: 0, 1: 1, 2:3}, 1: {0: 1, 1: 0, 2:2}, 2: {0: 3, 1:2, 2:0}}

    Note
    ----
    Uses pandas to accomplish this in a one liner.

    See Also
    --------
    dict2matrix

    """
    return pd.DataFrame(m).to_dict()


def dict2sparse(d):
    """
    Tranforms a 2D dictionary into a Scipy sparse matrix.

    Parameters
    ----------
    d : dict
        2D dictionary

    Returns
    -------
    m : CSR matrix
        CRS Sparse Matrix

    Examples
    --------
    >>> d = {0: {0: 0, 1: 1, 2:3}, 1: {0: 1, 1: 0, 2:2}, 2: {0: 3, 1:2, 2:0}}
    >>> dict2sparse(d)
        (0, 1)    1
        (0, 2)    3
        (1, 0)    1
        (1, 2)    2
        (2, 0)    3
        (2, 1)    2

    Note
    ----
    Uses pandas to convert dict into dataframe and then feeds it to the `csr_matrix`.

    See Also
    --------
    dict2matrix
    matrix2dict

    """
    return csr_matrix(pd.DataFrame.from_dict(d, orient='index').values)


def from_networkx_to_dijkstra_format(D, weight='weight'):
    """
    Converts a ``NetworkX.Graph`` object to input variables to be used by ``cython.dijkstra``.

    Parameters
    ----------
    D : NetworkX:Graph
        The Distance graph.

    weight : string
        The edge property to use as distance weight.

    Returns
    -------
    nodes : list
        List of all nodes converted to sequential numbers.

    edges : list
        List of all edges.

    neighbors : dict
        Dictionary containing the neighborhood of every node in a fast access format.

    dict_int_nodes : dict
        The mapping between original node names and the numeric node names.


    Examples
    --------
    >>> G = nx.path(5)
    >>> nx.set_edge_attributes(G, name='distance', values=1)
    >>> nodes, edges, neighbors, dict_int_nodes = from_networkx_to_dijkstra_format(G, weight='distance')
    """
    if not isinstance(D, nx.classes.graph.Graph):
        raise NotImplementedError("This is on the TODO list. For now, only undirected nx.Graphs() are accepted.")

    dict_nodes_int = {u: i for i, u in enumerate(D.nodes())}
    dict_int_nodes = {i: u for u, i in dict_nodes_int.items()}

    nodes = list(dict_nodes_int.values())

    edges_ij = {(dict_nodes_int[i], dict_nodes_int[j]): d[weight] for i, j, d in D.edges(data=True)}
    edges_ji = {(dict_nodes_int[j], dict_nodes_int[i]): d[weight] for i, j, d in D.edges(data=True)}

    edges = {**edges_ij, **edges_ji}

    neighbors = {dict_nodes_int[i]: [dict_nodes_int[j] for j in D.neighbors(i)] for i in D.nodes()}

    return nodes, edges, neighbors, dict_int_nodes

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
