from distanceclosure.dijkstra import Dijkstra
from distanceclosure._dijkstra import _py_single_source_shortest_distances  # , _py_single_source_complete_paths
from distanceclosure.cython._dijkstra import _cy_single_source_shortest_distances  # , _cy_single_source_complete_paths
#
from distanceclosure.utils import prox2dist
import numpy as np
import networkx as nx
#
# Test
#
# Numpy
P = np.array([
             [1., .9, .1, 0.],
             [.9, 1., .8, 0.],
             [.1, .8, 1., .6],
             [0., 0., .6, 1.]], dtype=float)
D = prox2dist(P)
#
# Edgelist
#
edgelist_luis = {
    ('s', 'b'): .9,
    ('s', 'c'): .1,
    ('b', 'c'): .8,
    ('c', 'd'): .6,
}

edgelist_james = {
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


# Single Source Shortest Paths > from Numpy
def test_dijkstra_sssd_python_from_numpy():
    """ Test Dijkstra: Single Source Shortest Path (SSSP) from Numpy """
    dij = Dijkstra.from_numpy_matrix(D, verbose=True)
    dij.single_source_shortest_distances(source=0, kind='metric', engine='python')


# Single Source Shortest Paths > from edgelist
def test_dijkstra_sssd_python_from_edgelist():
    """ Test Dijkstra: Single Source Shortest Path (SSSP) from Edgelist """
    dij = Dijkstra.from_edgelist(edgelist_james, verbose=True)
    dij.single_source_shortest_distances(source='s', kind='metric', engine='python')


def test_dijkstra_vs_networkx_single_source_all_lenghts_and_paths():
    """ Test Dijkstra: Rion's implementation vs Networkx implementation > Single Source """
    # NX Version
    G = nx.from_edgelist(edgelist_james)
    nx.set_edge_attributes(G, name='weight', values=edgelist_james)
    nx_lenghts = nx.single_source_dijkstra_path_length(G, source='s', weight='weight')
    nx_paths = nx.single_source_dijkstra_path(G, source='s', weight='weight')

    # My Version
    d = Dijkstra.from_edgelist(edgelist_james, directed=False)
    dc_lenghts = d.single_source_shortest_distances('s', kind='metric', engine='python')
    #dc_paths = d.single_source_shortest_paths('s')

    assert (nx_lenghts == d.get_shortest_distances(source='s', translate=True))
    #assert (nx_paths == d.get_shortest_paths(source='s', translate=True))


def test_dijkstra_vs_networkx_apsp():
    """ Test Dijkstra: Rion's implementation vs Networkx implementation > All Pairs """
    # NX Version
    G = nx.from_edgelist(edgelist_james)
    nx.set_edge_attributes(G, name='weight', values=edgelist_james)
    nx_all_lenghts = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    nx_all_paths = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))

    # My Version
    d = Dijkstra.from_edgelist(edgelist_james, directed=False)
    dc_all_lenghts = d.all_pairs_shortest_distances(n_jobs=1, kind='metric', engine='python')
    #dc_all_paths = d.all_pairs_shortest_paths(n_jobs=2, engine='python')

    assert (nx_all_lenghts == d.get_shortest_distances(translate=True))
    #assert (nx_all_paths == d.get_shortest_paths(translate=True))

def test_cython_dikjstra_vs_networkx_apsp()