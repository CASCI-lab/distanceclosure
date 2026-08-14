"""
Registries
==========

Mappings of arguments to their internal functionality.
"""

from ._backbone import _iterative_backbone, _flagged_backbone, _closure_backbone, _heuristic_backbone, _approximate_backbone, _drastic_disjunction
from ._closure import _dijkstra_closure

_KINDS = {
    "metric": sum,
    "ultrametric": max,
    "drastic": _drastic_disjunction
}

_BACKBONE_ALGORITHMS = {
    "iterative": _iterative_backbone,
    "flagged": _flagged_backbone,
    "closure": _closure_backbone,
    "heuristic": _heuristic_backbone,
    "approximate": _approximate_backbone
}

_CLOSURE_ALGORITHMS = {
    "dijkstra": _dijkstra_closure
}