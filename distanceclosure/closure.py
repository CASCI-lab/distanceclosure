# -*- coding: utf-8 -*-
"""
Transitive Closure
==================

Computes transitive closure on a weighted graph.
These algorithms work with undirected weighted (distance) graphs.
"""

from ._registries import _KINDS, _CLOSURE_ALGORITHMS

__all__ = [
    "distance_closure",
]

def distance_closure(D, kind='metric', algorithm='dijkstra', weight='weight', only_backbone=False, verbose=False, *args, **kwargs): 
    if kind == 'metric':
        disjunction = sum
    elif kind == 'ultrametric':
        disjunction = max

    if kind not in _KINDS:
        raise ValueError("Invalid input. Valid arguments are 'metric' and 'ultrametric'.")

    try:
        chosen_algorithm = _CLOSURE_ALGORITHMS[algorithm]
    except KeyError:
        raise ValueError("Invalid input. Valid arguments are 'dijkstra'")

    return chosen_algorithm(D, kind=kind, disjunction=disjunction, weight=weight, only_backbone=only_backbone, verbose=verbose, *args, **kwargs)
