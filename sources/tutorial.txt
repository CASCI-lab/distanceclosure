Tutorials
=========

Below are some examples of how you might use this package.

Backbone of Knowledge Networks
---------------------------------

If you have, let's say, a bipartite adjacency matrix ``X``.
The rows can be `users` and the columns are `terms`.
Each cell contains the number of times user ``i`` used term ``j``.
You want to know how the terms relate to each other.
Then you want to calculate pairwise distance between all columns, extract the metric and ultrametric backbone of this new ``knowledge network`` in the following way.


.. code-block:: python

	from distanceclosure.utils import prox2dist, dist2prox
	from distanceclosure.distance import pairwise_proximity
	from distanceclosure.closure import transitive_closure
	from distanceclosure.backbone import backbone
	import numpy as np

	X = np.array([
		[4,3,2,1],
		[3,2,1,0],
		[2,1,0,0],
		[1,0,0,0]])

	# Calculate Proximity and convert to Distance
	P = pairwise_proximity(X.T, metric='jaccard')
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
