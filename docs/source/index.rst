Distance Closure: the distance backbone of complex networks
============================================================

Description
************

This package implements methods for the calculation of the `Distance Closure` on Complex Networks, including its `metric` and `ultrametric` backbone.

The distance backbone is a principled graph reduction technique. It is a small subgraph sufficicent to compute all shortest paths. This method is suited for undirected weighted graphs, also known as proximity (similarity) or distance (dissimilarity) graphs.


Quick Install
--------------

.. code-block:: bash

    $pip install distanceclosure

Simple usage
-------------
    
.. code-block:: python
    :emphasize-lines: 2, 23

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
    # Make sure there an edge attribute with the distance value
    nx.set_edge_attributes(G, name='distance', values=edgelist)

    # Compute closure (note this will be a fully connected graph. It can be slow for large graphs)
    C = dc.distance_closure(G, kind='metric', weight='distance')

    # You can now access the new `metric_distance` value and whether the edge is part of the metric backbone.
    C['s']['c']
    > {'distance': 6, 'metric_distance': 6, 'is_metric': True}

If you are only interested in the metric backbone, you might want to only include distance values for edges already in the graph.

.. code-block:: python
    :emphasize-lines: 1

    C2 = dc.distance_closure(G, kind='metric', weight='distance', only_backbone=True)

    C.number_of_edges()
    > 22
    C2.number_of_edges()
    > 11

Formal definition
------------------

For the formal definition of the distance backbone, please refer to

- :cite:`Simas:2021` Tiago Simas, Rion Brattig Correia, and Luis M. Rocha. `The distance backbone of complex networks`__. *Journal of Complex Networks*, 9:cnab021, 2021. doi:10.1093/comnet/cnab021.

__ http://doi.org/10.1093/comnet/cnab021

- :cite:`Simas:2015` Tiago Simas and Luis M. Rocha. `Distance closures on complex networks`__. *Network Science*, 3:227–268, 6 2015. doi:10.1017/nws.2015.11.

__ http://doi.org/10.1017/nws.2015.11


Recent papers from our group that have used or built upon this method:

- :cite:`Correia:2022:meionav` Rion Brattig Correia, J.M. Almeida, M. Wyrwoll, I. Julca, D. Sobral, C.S. Misra, L.G. Guilgur, H. Schuppe, N. Silva, P. Prudêncio, A. Nóvoa, A.S. Leocádio, J. Bom, M. Mallo, S. Kliesch, M. Mutwil, Luis M. Rocha, F. Tüttelmann, J.D. Becker, and Paulo Navarro-Costa. `An old transcriptional program in male germ cells uncovers new causes of human infertility`__. **Under review**, 2022. doi:10.1101/2022.03.02.482557v2.

__ http://doi.org/10.1101/2022.03.02.482557v2

- :cite:`Correia:2022:contact` Rion Brattig Correia, Alain Barrat, and Luis M. Rocha. `The metric backbone preserves community structure and is a primary transmission subgraph in contact networks`__. **Under review**, 2022. doi:10.1101/2022.02.02.478784

__ http://doi.org/10.1101/2022.02.02.478784.

- :cite:`Correia:2016` Rion Brattig Correia, Lang Li, and Luis M. Rocha. `Monitoring potential drug interactions and reactions via network analysis of instagram user timelines`__. In *Pacific Symposium on Biocomputing*, volume 21, pages 492–503. 2016. doi:10.1142/9789814749411_0045.
    
__ http://www.informatics.indiana.edu/rocha/publications/PSB2016.php


Additional papers :cite:`Rocha:2002,Rocha:2005,Simas:2012,Ciampaglia:2015,Simas:2015,Correia:2016,Simas:2021, Correia:2022:contact,Correia:2022:meionav` can be found in the :doc:`bibliography` page.

Citation
---------

.. code-block:: bib

    @article{Simas:2021,
        author = {Tiago Simas and Rion Brattig Correia and Luis M. Rocha},
        doi = {10.1093/comnet/cnab021},
        issue = {6},
        journal = {Journal of Complex Networks},
        pages = {cnab021},
        title = {The distance backbone of complex networks},
        volume = {9},
        year = {2021}
    }


Documentation:
===================

.. toctree::
    :maxdepth: 3
    :caption: Table of Contents:

    install
    tutorial
    reference/index
    development
    bibliography


Indices and tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

