# -*- coding: utf-8 -*-
"""
Dijkstra: shortest path algorithm for weighted graphs
=====================================================

These methods are used for the computation of shortest paths on weighted graphs and were adapted from the Networkx (``networkx.algorithms.shortest_paths.weighted``) implementation.
"""

from heapq import heappush, heappop
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
__name__ = 'distanceclosure'
__author__ = """\n""".join(['Rion Brattig Correia <rionbr@gmail.com>'])

__all__ = [
    "all_pairs_dijkstra_path_length",
    "single_source_dijkstra_path_length"
]

def all_pairs_dijkstra_path_length(G, weight="weight", disjunction=sum):
    """Compute shortest path lengths between all nodes in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

    disjunction: function (default=sum)
        Whether to sum paths or use the max value.
        Use `sum` for metric and `max` for ultrametric.

    Returns
    -------
    distance : iterator
        (source, dictionary) iterator with dictionary keyed by target and
        shortest path length as the key value.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length = dict(all_pairs_dijkstra_path_length(G))
    >>> for node in [0, 1, 2, 3, 4]:
    ...     print(f"1 - {node}: {length[1][node]}")
    1 - 0: 1
    1 - 1: 0
    1 - 2: 1
    1 - 3: 2
    1 - 4: 3
    >>> length[3][2]
    1
    >>> length[2][2]
    0

    Note
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The dictionary returned only has keys for reachable node pairs.
    """
    weight_function = _weight_function(G, weight)
    for n in G:
        yield (n, single_source_dijkstra_path_length(G, source=n, weight_function=weight_function, disjunction=disjunction))


def single_source_dijkstra_path_length(G, source, weight_function, paths=None, disjunction=sum):
    """Uses (a custom) Dijkstra's algorithm to find shortest weighted paths

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node for path.

    weight_function: function
        Function with (u, v, data) input that returns that edges weight

    paths: dict, optional (default=None)
        dict to store the path list from source to each node, keyed by node.
        If None, paths are not stored.

    disjunction: function (default=sum)
        Whether to sum paths or use the max value.
        Use `sum` for metric and `max` for ultrametric.

    Returns
    -------
    distance : dictionary
        A mapping from node to shortest distance to that node from one
        of the source nodes.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Note
    -----
    The optional predecessor and path dictionaries can be accessed by
    the caller through the original paths objects passed
    as arguments. No need to explicitly return paths.

    """
    G_succ = G._succ if G.is_directed() else G._adj

    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    seen = {}
    # fringe is heapq with 3-tuples (distance,c,node)
    # use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []
    if source not in G:
        raise nx.NodeNotFound(f"Source {source} not in G")
    seen[source] = 0
    push(fringe, (0, next(c), source))
    while fringe:
        (d, _, v) = pop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        for u, e in G_succ[v].items():
            cost = weight_function(v, u, e)
            if cost is None:
                continue
            vu_dist = disjunction([dist[v], cost])
            if u in dist:
                u_dist = dist[u]
                if vu_dist < u_dist:
                    raise ValueError("Contradictory paths found:", "negative weights?")
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, next(c), u))
                if paths is not None:
                    paths[u] = paths[v] + [u]
    #
    return dist
