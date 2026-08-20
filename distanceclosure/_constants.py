from distanceclosure.backbone import _drastic_disjunction

_KINDS = {
    "metric": sum,
    "ultrametric": max,
    "drastic": _drastic_disjunction
}