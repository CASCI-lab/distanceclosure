Distance Closure on Complex Networks (Backbone Extraction)
============================================================

Description:
-------------

This package implements some of the methods for the calculation of Distance Closure on Complex Networks. 
The distance closure can be calculated using different metrics. Currently supported are 'Metric' and 'Ultrametric'.

The mathematical description of the methods can be seen in:

	"T. Simas and L.M. Rocha [2015]."[Distance Closures on Complex Networks](http://www.informatics.indiana.edu/rocha/publications/NWS14.php)". Network Science, 3(2):227-268. doi:10.1017/nws.2015.11"

Installation:
---------------

This package is available on Pypi. Starting on version 0.3.6 (*with the addition of the _Dijkstra module in Cython*), if you are on a Mac, you can just install it directly from Pypi with the following command on terminal.

```
$pip install distanceclosure
```

If you are on Linux or Windows you might need to download the package either from the Pypi website or clone the [project page](https://github.com/rionbr/distanceclosure) repository and install the package from source. For example:

```
> git clone https://github.com/rionbr/distanceclosure.git
> cd distanceclosure
> python setup.py install
```

Docs:
------

The full documentation can be found at: [rionbr.github.io/distanceclosure](https://rionbr.github.io/distanceclosure)