# -*- coding: utf-8 -*-
"""
Transitive Closure
==================

Computes transitive closure on a weighted graph.
These algorithms work with undirected weighted (distance) graphs.
"""

import numpy as np
from typing import Callable
from ._constants import _KINDS
from distanceclosure._dijkstra import _all_pairs_dijkstra_path_length


__all__ = [
    "distance_closure",
]

def distance_closure(D, kind='metric', weight='weight', only_backbone=False, verbose=False, *args, **kwargs): 
    if kind == 'metric':
        disjunction = sum
    elif kind == 'ultrametric':
        disjunction = max

    if kind not in _KINDS:
        raise ValueError("Invalid input. Valid arguments are 'metric' and 'ultrametric'.")

    return _closure(D, kind=kind, disjunction=disjunction, weight=weight, only_backbone=only_backbone, verbose=verbose, *args, **kwargs)


# Private
def _closure(D, kind: str, disjunction: Callable, weight: str, only_backbone: bool, verbose: bool, *args, **kwargs):
    G = D.copy() 

    edges_seen = set()
    i = 1
    total = G.number_of_nodes()
    # APSP
    for u, lengths in _all_pairs_dijkstra_path_length(G, weight=weight, disjunction=disjunction):
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
                    if not only_backbone:
                        G.add_edge(u, v, **{weight: np.inf, kind_distance: length})
                else:
                    G[u][v][kind_distance] = length
                    G[u][v][is_kind] = True if (length == G[u][v][weight]) else False
        i += 1

    return G
