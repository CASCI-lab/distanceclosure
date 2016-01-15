Distance Closure on Complex Networks – Backbone
============================================================

Description:
-------------

This package implements some of the methods for the calculation of Distance Closure on Complex Networks. 
The distance closure can be calculated using different metrics. Currently supported are 'Metric' and 'Ultrametric'.

The mathematical description of the methods can be seen in:

	"T. Simas and L.M. Rocha [2015]."`Distance Closures on Complex Networks`__". *Network Science*, **3** (2):227-268. doi:10.1017/nws.2015.11"

	__ http://www.informatics.indiana.edu/rocha/publications/NWS14.php

Other papers which have used this method:

	"R.B. Correia, L. Li, L.M. Rocha [2016]. "`Monitoring potential drug interactions and reactions via network analysis of Instagram user timeliness`__". *Pacific Symposium on Biocomputing*. **21**:492-503."
	
	__ http://www.informatics.indiana.edu/rocha/publications/PSB2016.php

Installation:
---------------

This package is available on Pypi. Just run the following command on terminal to install.

.. code-block:: bash

	$pip install distance_closure

You can also source the code directly from the github `project page`__.

__ https://github.com/rionbr/distanceclosure

Usage:
--------

If you have, let's say, a bipartite adjacency matrix ``X``.
You want to calculate pairwise distance between all columns and then extract the metric and ultrametric backbone of this new network, you would do the following.


.. code-block:: python

	from distanceclosure.utils import prox2dist, dist2prox
	from distanceclosure.closure import transitive_closure
	from distanceclosure.backbone import backbone

	# Calculate Proximity and convert to Distance
	P = pairwise_proximity(X, metric='jaccard')
	D = prox2dist(P)

	# Calculate transitive closure using the metric and ultra-metric measure
	Cm = transitive_closure(D, kind='metric')
	Cu = transitive_closure(D, kind='ultrametric')

	# Retrieve the backbone edges
	Bm = backbone(D, Cm)
	Bu = backbone(D, Cu)


The backbone edges on ``Bm`` and ``Bu`` can be accessed using 

.. code-block:: python

	import numpy as np
	rows, cols = np.where(Bm==1)

where edges with a ``1`` are metric, ``2`` are semi-metric and ``0`` are non-existent. The diagonal is ``-1``.

Methods:
--------
.. toctree::
	:maxdepth: 2

	distance
	closure
	backbone
	utils

Notes:
-------

The current version of this code cannot handle extra large networks since it uses dense matrix computation.
Pull requests are welcome.

Here are some TODOs:

* Transitive closure (with custom distance metric) using the dijkstra algorithm;
* Porting to Cython.

Pull requests are welcome. Please get in contact with me beforehand ``rionbr a@t gmail d.t com``.

Tests:
--------
Run ``nosetests -v`` to perform tests and diagnoses on functions.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

