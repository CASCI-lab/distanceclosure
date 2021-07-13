Distance Closure on Complex Networks (Network Backbone)
=======================================================


Description
-------------

This package implements methods to calculate the Distance Closure of Complex Networks including its Metric and UltraMetric Backbone.


Installation
---------------

Latest development release on GitHub

```
pip install git+git://github.com/rionbr/distanceclosure
```

Latest PyPI stable release:

```
$pip install distanceclosure
```


Docs
-----

The full documentation can be found at: [rionbr.github.iodistanceclosure/](https://rionbr.github.io/distanceclosure)


Papers
-------

- T. Simas, R.B. Correia, L.M. Rocha [2021]. "[The distance backbone of complex networks](https://arxiv.org/abs/2103.04668)". *Journal of Complex Networks*. In Press. arXiv:2103.04668

- T. Simas and L.M. Rocha [2015]."[Distance Closures on Complex Networks](http://www.informatics.indiana.edu/rocha/publications/NWS14.php)". *Network Science*, **3**(2):227-268. doi:10.1017/nws.2015.11


Credits
--------

``distanceclosure`` was originally written by Rion Brattig Correia with input from many others. Thanks to everyone who has improved ``distanceclosure`` by contributing code, bug reports (and fixes), documentation, and input on design, and features.


Support
-------

Those who have contributed to ``distanceclosure`` have received support throughout the years from a variety of sources.  We list them below.

- [CASCI](https://homes.luddy.indiana.edu/rocha/casci.php), Indiana University, Bloomington, IN; PI: Luis M. Rocha
- [CAPES Foundation](https://www.gov.br/capes/pt-br), Ministry of Education of Brazil, Brasília, Brazil; Rion B. Correia.


Development
-----------
Pull requests are welcome :) Please get in touch beforehand: `rionbr(at)gmail(dot)com`.


Changelog
---------

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