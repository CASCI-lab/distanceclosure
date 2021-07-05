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

__author__ = """\n""".join([
    'Rion Brattig Correia <rionbr@gmail.com>',
    'Luis Rocha <rocha@indiana.com>'])

__all__ = ['prox2dist',
           'dist2prox',
           'dict2matrix',
           'matrix2dict',
           'dict2sparse',
           'from_networkx_to_dijkstra_format']


def prox2dist(p):
    """
    Transforms a non-negative ``[0,1]`` proximity to distance in the ``[0,inf]`` interval:

    .. math::

        d = \\frac{1}{p} - 1

    Args:
        p (float): proximity value

    Returns:
        d (float): distance value

    See Also:
        :attr:`dist2prox`
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

    Args:
        D (matrix): Distance matrix

    Returns:
        P (matrix): Proximity matrix

    See Also:
        :attr:`prox2dist`

    """
    if d == np.inf:
        return 0
    else:
        return (d + 1) ** -1


def dict2matrix(d):
    """
    Tranforms a 2D dictionary into a numpy. Usefull when converting Dijkstra results.

    Args:
        d (dict): 2D dictionary

    Returns:
        m (matrix): numpy matrix

    Warning:
        If your nodes have names instead of number assigned to them, make sure to keep a mapping.

    Usage:
        >>> d = {0: {0: 0, 1: 1, 2:3}, 1: {0: 1, 1: 0, 2:2}, 2: {0: 3, 1:2, 2:0}}
        >>> dict2matrix(d)
        [[ 0 1 3]
         [ 1 0 2]
         [ 3 2 0]]

    See Also:
        :attr:`matrix2dict`

    Note:
        Uses pandas to accomplish this in a one liner.
    """
    return pd.DataFrame.from_dict(d).values


def matrix2dict(m):
    """
    Tranforms a Numpy matrix into a 2D dictionary. Usefull when comparing dense metric and Dijkstra results.

    Args:
        m (matrix): numpy matrix

    Returns:
        d (dict): 2D dictionary

    Usage:
        >>> m = [[0, 1, 3], [1, 0, 2], [3, 2, 0]]
        >>> matrix2dict(m)
        {0: {0: 0, 1: 1, 2:3}, 1: {0: 1, 1: 0, 2:2}, 2: {0: 3, 1:2, 2:0}}

    See Also:
        :attr:`dict2matrix`

    Note:
        Uses pandas to accomplish this in a one liner.
    """
    return pd.DataFrame(m).to_dict()


def dict2sparse(d):
    """
    Tranforms a 2D dictionary into a Scipy sparse matrix.

    Args:
        d (dict): 2D dictionary

    Returns:
        m (csr matrix): CRS Sparse Matrix

    Usage:
        >>> d = {0: {0: 0, 1: 1, 2:3}, 1: {0: 1, 1: 0, 2:2}, 2: {0: 3, 1:2, 2:0}}
        >>> dict2sparse(d)
        (0, 1)    1
        (0, 2)    3
        (1, 0)    1
        (1, 2)    2
        (2, 0)    3
        (2, 1)    2

    See Also:
        :attr:`dict2matrix`, :attr:`matrix2dict`

    Note:
        Uses pandas to convert dict into dataframe and then feeds it to the `csr_matrix`.
    """
    return csr_matrix(pd.DataFrame.from_dict(d, orient='index').values)


def from_networkx_to_dijkstra_format(G, weight='weight'):
    """
    Converts a `networkx.Graph` object to the a custom dijkstra format used in `cython.dijkstra`.

    Args:
        G (networkx.Graph) : Distance graph edgelist distance adjacency matrix.
        weight (string) : The edge property to use as distance weight.

    Returns:
        nodes (list), edges (list), neighbors (dict): tuple of variables.

    Examples:
        >>> G = nx.path(5)
        >>> nx.set_edge_attributes(G, name='distance', values=1)
        >>> nodes, edges, neighbors = from_networkx_to_dijkstra_format(G, weight='distance')
    """
    if type(G) != nx.classes.graph.Graph:
        raise NotImplementedError("This is on the TODO list. For now, only undirected nx.Graphs() are accepted.")

    dict_nodes_int = {u: i for i, u in enumerate(G.nodes())}

    nodes = list(dict_nodes_int.values())

    edges_ij = {(dict_nodes_int[i], dict_nodes_int[j]): d[weight] for i, j, d in G.edges(data=True)}
    edges_ji = {(dict_nodes_int[j], dict_nodes_int[i]): d[weight] for i, j, d in G.edges(data=True)}

    edges = {**edges_ij, **edges_ji}

    neighbors = {dict_nodes_int[i]: [dict_nodes_int[j] for j in G.neighbors(i)] for i in G.nodes()}

    return nodes, edges, neighbors
