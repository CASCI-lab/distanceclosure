"""
Registries
==========

Mappings of arguments to their internal functionality.
"""

from ._algorithms import _iterative_backbone, _flagged_backbone, _closure_backbone, _heuristic_backbone, _approximate_backbone, _drastic_disjunction

_KINDS = {
    "metric": sum,
    "ultrametric": max,
    "drastic": _drastic_disjunction
}

_ALGORITHMS = {
    "iterative": _iterative_backbone,
    "flagged": _flagged_backbone,
    "closure": _closure_backbone,
    "heuristic": _heuristic_backbone,
    "approximate": _approximate_backbone
}
