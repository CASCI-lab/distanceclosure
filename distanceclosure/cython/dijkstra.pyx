# -*- coding: utf-8 -*-
"""
Transitive closure
===================

Computes transitive closure on a graph Adjacency Matrix.

These algorithms work with undirected weighted (distance) graphs.
"""
#    Copyright (C) 2021 by
#    Rion Brattig Correia <rionbr@gmail.com>
#    All rights reserved.
#    MIT license.
from libc.stdlib cimport malloc, free
from libc.float cimport DBL_MAX
from libc.stdio cimport printf
cimport distanceclosure.cython.pqueue
from distanceclosure.cython.pqueue cimport *
#
__author__ = """\n""".join(['Rion Brattig Correia <rionbr@gmail.com>'])


def cy_all_pairs_dijkstra_path_length(list N, dict E, dict neighbors, tuple operator_names):
    """
    Computes shortest path between all pairs of nodes (APSP).

    Args:
        N (list): the set of nodes in the network.
        E (dict): the dict of edges in the network.
        neighbors (dict): a dict that contains all node neighbors.
        operators (tuple): a tuple of the operators to compute shortest path. Default is ``(min,max)``.

    Returns:
        source, distances (tuple iterator): A tuple iterator is returned for every node and its distances to every other node.
    """
    for source in N:
        yield (source, cy_single_source_dijkstra_path_length(source, N, E, neighbors, operator_names))


def cy_single_source_dijkstra_path_length(int source, list N, dict E, dict neighbors, tuple operator_names):
    """
    The Cython wrapper for the C function `_c_single_source_shortest_distances`
    """
    return c_single_source_dijkstra_path_length(source, N, E, neighbors, operator_names)


cdef dict c_single_source_dijkstra_path_length(int source, list N, dict E, dict neighbors, tuple operator_names):
    """
    Compute shortest path between source and all other reachable nodes.

    Args:
        source (int): the source node.
        N (list): the set of nodes in the network.
        E (dict): the dict of edges in the network.
        neighbors (dict): a dict that contains all node neighbors.
        operators (tuple): a tuple of the operators to compute shortest path. Default is ``(min,max)``.

    Returns:
        dists (dict): the final distance calculated from source to all other nodes.
        paths (dict): the final path between source and all other nodes.

    Note:
        The python `heapq` module does not support item update.
        Therefore this algorithm keeps track of which nodes and edges have been searched already;
        and the queue itself has duplicated nodes inside.
    """
    #
    # Variable Definition
    #
    cdef pqueue_t *Q        # The Priority Queue
    cdef node_t *n          # A Data struct (node_t) to be manipulated pushed and popped from the PQueue
    cdef dict final_dist    # Final distance from source to every other node
    #cdef dict paths         # The paths
    cdef set visited_edges  # Visited Edges
    cdef int i              # indices
    cdef int n_nodes        # Number of nodes
    #
    # Variable Init
    #
    # New Queue
    Q = pqueue_init(10, cmp_pri, get_pri, set_pri, get_pos, set_pos)
    if not Q:
        raise MemoryError()
    #
    final_dist = dict()
    #paths = dict()
    visited_edges = set()
    n_nodes = len(N)
    # Operators
    conj, disj = operator_names
    if disj == 'sum':
        disj = sum
    elif disj == 'max':
        disj = max

    # The Data structs (node_t) to be added to the PQueue
    cdef node_t *ns = <node_t *>malloc(n_nodes * sizeof(node_t))
    if not ns:
        raise MemoryError()

    # Populate Priority Queue with nodes and infinity distance
    for i, node in enumerate(N, start=0):
        if node == source:
            # Source node receives distance zero
            final_dist[source] = 0
            # PQ value
            ns[i].node = i
            ns[i].dist = 0
            # Add to PQ
            pqueue_insert(Q, &ns[i])
        # All other nodes receive distance infinity (LLONG_MAX priority)
        else:
            final_dist[node] = float('inf')
            # PQ values
            ns[i].node = i
            ns[i].dist = DBL_MAX
            # Add to PQ
            pqueue_insert(Q, &ns[i])

    # Loop the PQ
    while pqueue_size(Q):
        n = <node_t*>pqueue_pop(Q)

        # Iterate over all neighbors of node 
        for neighbor in neighbors[n.node]:

            # If this edge has been searched, continue
            if (n.node, neighbor) in visited_edges:
                continue

            # the edge distance/weight/cost
            weight = E[(n.node, neighbor)]

            # Operation to decide how to compute the lenght, summing edges (metric) or taking the max (ultrametric)
            new_dist = disj([n.dist, weight])

            # If this is a shortest distance, update
            if new_dist < final_dist[neighbor]:
                # update the shortest distance to this node
                final_dist[neighbor] = new_dist
                # update this neighbor on the queue
                pqueue_change_priority(Q, <pqueue_pri_t>new_dist, &ns[neighbor])
                # update the path
                #paths[neighbor] = [new_dist, n.node]

            # Add to visited edges
            visited_edges.add((neighbor, n.node))

    # Free the Memory used
    pqueue_free(Q)
    free(ns);

    # Return
    return final_dist  # , paths


def cy_single_source_complete_paths(int source, list N, dict paths):
    """
    The Cython wrapper for the C function `_c_single_source_shortest_paths`
    """
    return c_single_source_shortest_paths(source, N, paths)


cdef c_single_source_shortest_paths(int source, list N, dict paths):
    """
    From the dict of node parent paths, recursively return the complete path from the source to all targets

    Args:
        source (int): the source node.
        N (dict): the set of nodes in the network.
        paths (dict): a dict of nodes and their distance to the parent node.

    Returns:
        path (dict): the complete path between source and all other nodes, including source and target in the list.

    """
    # Init
    cdef int n
    cdef list plist
    cdef dict complete_paths
    # Method
    complete_paths = dict()
    for n in N:
        plist = __c__get_path_recursive(paths, [], source, n)
        plist.append(source)
        plist.reverse()
        complete_paths[n] = plist

    return complete_paths


cdef __c__get_path_recursive(dict paths, list plist, int source, int n):
    """
    Recursive function to traverse the nodes between source and 
    """
    if n != source:
        plist.append(n)
        try:
            n = paths[n][1]
        except:
            pass
        else:
            __c__get_path_recursive(paths, plist, source, n)
    return plist


