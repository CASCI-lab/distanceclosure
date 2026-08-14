# -*- coding: utf-8 -*-
"""
Distance Backbones
==================

Compute the distance backbones of weighted graphs.
"""
import networkx as nx
from ._registries import _BACKBONE_ALGORITHMS, _KINDS

__all__ = [
    "distance_backbone",
    "metric_backbone",
    "ultrametric_backbone" 
]

def distance_backbone(D: nx.Graph | nx.DiGraph, weight: str = "weight", kind: str = "metric", algorithm: str = "iterative", distortion: bool = False, self_loops: bool = False, cutoff: int = None, verbose: bool = False, *args, **kwargs) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
    """
    Compute the distance backbone of a weighted graph.

    Parameters
    ----------
    D : Directed or undirected NetworkX graph
        A weighted distance graph
    weight : str, optional
        Edge property containing distance values, by default 'weight'
    kind : {"metric", "ultrametric"}, optional
        Distance metric used to compute the backbone. "metric" uses sum and "ultrametric" uses max, by default "metric".
    algorithm : {"iterative", "flagged", "closure", "heuristic", "approximate"}, optional
        Algorithm used to compute the backbone, by default "iterative".
    distortion : bool, optional
        Whether to compute edge distortion from edges not in backbone, by default False
    self_loops : bool, optional
        If the distance graph has nodes with self distance greater than zero, by default False
    cutoff : int, optional
        Maximum number of connections in the path. If None, compute the entire closure as is the cutoff is the number of nodes, by default None
    verbose : bool, optional
        Whether to display progress information, by default False.
    *args
        Additional positional arguments passed to the selected algorithm.
    **kwargs
        Additional keyword arguments passed to the selected algorithm.

    Returns
    -------
    nx.Graph or nx.DiGraph
        The distance backbone.
    dict
        Edge distortions, returned with the backbone when ``distortion=True``.

    Raises
    ------
    ValueError
        If ``kind`` or ``algorithm`` is invalid.
    """

    if self_loops:
        raise NotImplementedError
    if cutoff is not None:
        raise NotImplementedError
    
    if kind == 'metric':
        disjunction = sum
    elif kind == 'ultrametric':
        disjunction = max
    elif kind == 'drastic':
        disjunction = _drastic_disjunction

    if kind not in _KINDS:
        raise ValueError("Invalid input. Valid arguments are 'metric' and 'ultrametric'.")

    try:
        chosen_algorithm = _BACKBONE_ALGORITHMS[algorithm]
    except KeyError:
        raise ValueError("Invalid input. Valid arguments are 'iterative', 'flagged', 'closure', 'heuristic', or 'approximate'")
    
    return chosen_algorithm(D, weight=weight, disjunction=disjunction, distortion=distortion, self_loops=self_loops, cutoff=cutoff, verbose=verbose, *args, **kwargs)


def metric_backbone(D: nx.Graph | nx.DiGraph, weight: str = "weight", distortion: bool = False, self_loops: bool = False, cutoff: int = None, verbose: bool = False, *args, **kwargs) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
    """
    Compute the metric backbone of a weighted graph.

    This is a wrapper for :func:`distance_backbone`
    where ``kind="metric"`` and ``algorithm="iterative"``.
    """

    return distance_backbone(D, weight=weight, algorithm="iterative", kind="metric", distortion=distortion, self_loops=self_loops, cutoff=cutoff, verbose=verbose, *args, **kwargs)


def ultrametric_backbone(D: nx.Graph | nx.DiGraph, weight: str = "weight", distortion: bool = False, self_loops: bool = False, cutoff: int = None, verbose: bool = False, *args, **kwargs) -> nx.Graph | nx.DiGraph | tuple[nx.Graph | nx.DiGraph, dict]:
    """
    Compute the metric backbone of a weighted graph.

    This is a wrapper for :func:`distance_backbone`
    where ``kind="ultrametric"`` and ``algorithm="iterative"``.
    """

    return distance_backbone(D, weight=weight, algorithm="iterative", kind="ultrametric", distortion=distortion, self_loops=self_loops, cutoff=cutoff, verbose=verbose, *args, **kwargs)
