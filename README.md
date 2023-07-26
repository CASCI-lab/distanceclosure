Distance Closure on Complex Networks (Network Backbone)
=======================================================


Description
-------------

This package implements methods to calculate the Distance Closure of Complex Networks including its Metric and UltraMetric Backbone.


Installation
---------------

Latest development release on GitHub (v0.5):

```
pip install git+https://github.com/CASCI-lab/distanceclosure
```

Latest PyPI stable release (v0.4.1):

```
$pip install distanceclosure
```


Simple usage
------------

How to calculate Closure of a weighted distance graph:

```python
    import networkx as nx
    import distanceclosure as dc

    # Instanciate a (weighted) graph
    edgelist = {
        ('s', 'a'): 8,
        ('s', 'c'): 6,
        ('s', 'd'): 5,
        ('a', 'd'): 2,
        ('a', 'e'): 1,
        ('b', 'e'): 6,
        ('c', 'd'): 3,
        ('c', 'f'): 9,
        ('d', 'f'): 4,
        ('e', 'g'): 4,
        ('f', 'g'): 0,
    }
    G = nx.from_edgelist(edgelist)
    # Make sure every edge has an attribute with the distance value
    nx.set_edge_attributes(G, name='distance', values=edgelist)

    # Compute closure (note this will be a fully connected graph for an undirected connected component. It can be slow for large graphs)
    C = dc.distance_closure(G, kind='metric', weight='distance')

    # You can now access the new `metric_distance` value and whether the edge is part of the metric backbone.
    C['s']['c']
    > {'distance': 6, 'metric_distance': 6, 'is_metric': True}
```

If you are only interested in the metric backbone itself:

```python
    B = dc.metric_backbone(G, weight='distance')

    B.number_of_edges()
    > 9
    C.number_of_edges()
    > 28
    G.number_of_edges()
    > 11
```

Papers
-------

- F.X. Costa, R.B. Correia, L.M. Rocha [2023]. "[The distance backbone of directed networks](https://link.springer.com/chapter/10.1007/978-3-031-21131-7_11)". In: Cherifi, H., Mantegna, R.N., Rocha, L.M., Cherifi, C., Micciche, S. (eds) *Complex Networks and Their Applications XI. COMPLEX NETWORKS 2022.* Studies in Computational Intelligence, vol 1078. Springer, Cham. doi: 10.1007/978-3-031-21131-7_11

- T. Simas, R.B. Correia, L.M. Rocha [2021]. "[The distance backbone of complex networks](https://academic.oup.com/comnet/article/9/6/cnab021/6403661)". *Journal of Complex Networks*, 9 (**6**):cnab021. doi: 10.1093/comnet/cnab021

- T. Simas and L.M. Rocha [2015]."[Distance Closures on Complex Networks](http://www.informatics.indiana.edu/rocha/publications/NWS14.php)". *Network Science*, 3(**2**):227-268. doi:10.1017/nws.2015.11


Credits
--------

``distanceclosure`` was originally written by Rion Brattig Correia with input from many others. Thanks to everyone who has improved ``distanceclosure`` by contributing code, bug reports (and fixes), documentation, and input on design, and features.


Support
-------

Those who have contributed to ``distanceclosure`` have received support throughout the years from a variety of sources.  We list them below.

- [CASCI](https://casci.binghamton.edu/casci.php), Binghamton University, Binghamton, NY; PI: Luis M. Rocha
- [CAPES Foundation](https://www.gov.br/capes/pt-br), Ministry of Education of Brazil, Brasília, Brazil; Rion B. Correia.


Development
-----------

Pull requests are welcome :) Please get in touch beforehand: `rionbr(at)gmail(dot)com` or `fcosta(at)binghamton(dot)edu`.


Changelog
---------

v0.5
- Iterative backbone computation (faster for graphs with smaller backbones)

v0.4
- Code simplification and compliance to NetworkX

v0.3.6
- Dijkstra Cythonized

v0.3.2
- S and B measure added to closure

v0.3.0
- Dijkstra APSP algorithm
- Support for sparse matrices

v0.2
- First docs released

v0.1
- First release nad dense matrix APSP