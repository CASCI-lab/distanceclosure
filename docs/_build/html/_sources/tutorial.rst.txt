.. role:: py(code)
   :language: python

Tutorials
=========

Below are some examples of how to transform your data into a ``proximity``/``distance`` weighted graph, and then extract the graph's ``metric`` and ``ultrametric`` backbones.


Loading packages
-----------------

You will most definetely need two packages to start: NetworkX (for handling networks), and the distanceclosure (to calculate their backbones).

.. code-block:: python

    import networkx as nx
    import distanceclosure as dc


Proximities and distances
--------------------------

Usually, there are two way to think about graph weights, one is consider every edge like a ``proximity`` (or a similarity), and the other is to think it in terms of a ``distance`` (or a dissimilarity).

In ``proximity`` graphs weights are like probabilities, they range between ``[0,1]``. The stronger (higher) the proximity, the more the nodes are alike, or similar. These graphs are usually obtained from computing the pairwise similarity among all pairs of variables (nodes).

In ``distance`` graphs, weight are (er) distances in a particular space (often not euclidean), and they range between ``[0, infinity]``. The smaller the distance, the closer together two nodes are in that space.


Now, proximity and distance measures are isomorphic and we can convert proximity into distances and vice versa.

.. code-block:: python
    :caption: Proximity to Distances
    
    >>> dc.utils.prox2dist(1)
    0
    >>> dc.utils.prox2dist(0)
    inf

.. code-block:: python
    :caption: Distances to Proximities
    
    >>> dc.utils.dist2prox(15)
    0.0625
    >>> dc.utils.dist2prox(1)
    0.5


Building a weighted graph and extracting the backbone
------------------------------------------------------

Let's say you observed a phenomenon :code:`n = 10` times. Think scientists talking during coffee breaks on a conference, co-expressed genes in replicate experiments, or friends seen together in photos. Each time, you ``counted`` how many times each pair of nodes (scientists, genes, friends) were observed together. If you translate these observations into an edgelist format, you have

.. code-block:: python

    >>> counts = {
        ('i', 'j'): 1,
        ('i', 'l'): 2,
        ('i', 'k'): 1,
        ('l', 'k'): 2,
        ('k', 'm'): 5,
        ('k', 'j'): 1,
        ('m', 'j'): 5,
    }

Normalizing the number of observations by the total number of events, you end up with a probability of interaction – or a proximity (similarity) measure. The higher the value, the higher the chances these two nodes were seen together.

.. code-block:: python

    >>> proximity = {ij: c/n for ij, c in counts.items()}
    >>> proximity
    {
        ('i', 'j'): 0.1,
        ('i', 'l'): 0.2,
        ('i', 'k'): 0.1,
        ('l', 'k'): 0.2,
        ('k', 'm'): 0.5,
        ('k', 'j'): 0.1,
        ('m', 'j'): 0.5,
    }

To convert this similarity into a distance, we use the ``prox2dist`` function.

.. code-block:: python

    >>> distance = {ij: dc.utils.prox2dist(p) for ij, p in proximity.items()}
    {
        ('i', 'j'): 9,
        ('i', 'l'): 4,
        ('i', 'k'): 9,
        ('l', 'k'): 4,
        ('k', 'm'): 1,
        ('k', 'j'): 9,
        ('m', 'j'): 1,
    }

Now we use NetworkX and convert this edgelist format into a undirected weighted distance NetworkX.Graph object.

.. code-block:: python

    >>> D = nx.from_edgelist(distance)
    # Make sure every edge has an attribute with the distance value
    >>> nx.set_edge_attributes(D, name='distance', values=edgelist)

And compute the ``metric`` and ``ultrametric`` backbone of this ``distance graph``.

.. code-block:: python

    >>> Dcm = dc.distance_closure(D, kind='metric', weight='distance')
    >>> Dcum = dc.distance_closure(D, kind='ultrametric', weight='distance')

.. note::

    ``distance_closure`` often returns a fully connected graph, which can be quite computationally expensive you have a large graph.
    If you are only interested in the metric or ultrametric backbone, you should only add new distance values for edges already in the graph, using

    .. code-block:: python
        
        Dcm = dc.distance_closure(D, kind='metric', weight='distance', only_backbone=True)

Metric edges are now identified with attributes ``metric_distance`` and ``is_metric``. Similarly, ultrametric edges are identified with attributes ``ultrametric_distance`` and ``is_ultrametric``.
To identify them, simply do

.. code-block:: python

    >>> [(i, j) for i, j, d in Dcm.edges(data=True) if d['is_metric'] is True]
    [('i', 'j'), ('i', 'l'), ('j', 'm'), ('l', 'k'), ('k', 'm')]
    
    >>> [(i, j) for i, j, d in Dcum.edges(data=True) if d['is_ultrametric'] is True]
    [('i', 'l'), ('j', 'm'), ('l', 'k'), ('k', 'm')]


You can also export the results to a ``Pandas.DataFrame`` with

.. code-block:: python

    >>> nx.to_pandas_edgelist(Dcm)
      source target  is_metric  distance  metric_distance
    0      i      j       True       9.0                9
    1      i      l       True       4.0                4
    2      i      k      False       9.0                8
    3      i      m      False       inf                9
    4      j      k      False       9.0                2
    5      j      m       True       1.0                1
    6      j      l      False       inf                6
    7      l      k       True       4.0                4
    8      l      m      False       inf                5
    9      k      m       True       1.0                1

This toy example is the same shown in :cite:`Simas:2021`, Figures 4 and 5.
