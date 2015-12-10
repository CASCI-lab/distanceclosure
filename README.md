Distance Closure on Complex Networks (Backbone)
===============================================

Description:
-----------

This package implements some of the methods for the calculation of Distance Closure on Complex Networks. 
It calculates network pairwise distance using the jaccard similarity/distance measure and transitive closure using both metric and ultrametric measures.

The mathematical description of the methods can be seen in:
```
T. Simas and L.M. Rocha [2015].\"Distance Closures on Complex Networks\". Network Science, 3(2):227-268. doi:10.1017/nws.2015.11
```

Usage:
------

```python
from distanceclosure import pairwise_proximity, prox2dist, transitive_closure, backbone

# Calculate Proximity and convert to Distance
P = pairwise_proximity(X, metric='jaccard')
D = prox2dist(P)

# Calculate transitive closure using the metric and ultra-metric measure
Cm = transitive_closure(D, kind='metric')
Cu = transitive_closure(D, kind='ultrametric')

# Retrieve the backbone edges
Bm = backbone(D, Cm)
Bu = backbone(D, Cu)
```

The backbone edges on `Bm` and `Bu` can be accessed using 
```python
rows, cols = np.where(Bm==1)
```
where edges with a `1` are metric, `2` are semi-metric and `0` are non-existent. The diagonal is `-1`.

Notes:
-----

The current version of this code cannot handle extra large networks since it uses matrix computation.

Pull requests are welcome. Here are some TODOs:
- Transitive closure using the dijkstra algorithm
- Porting to Cython

Tests:
------
Run `nosetests -v` to perform tests and diagnoses on functions.
