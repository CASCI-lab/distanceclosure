from distanceclosure.dijkstra import single_source_dijkstra_path_length
from distanceclosure.cython.dijkstra import cy_single_source_dijkstra_path_length
#
from distanceclosure.utils import prox2dist, from_networkx_to_dijkstra_format
import numpy as np
import networkx as nx
#
from networkx.algorithms.shortest_paths.weighted import _weight_function
#
# Test
#
# Numpy
P = np.array([
             [1., .9, .1, 0.],
             [.9, 1., .8, 0.],
             [.1, .8, 1., .6],
             [0., 0., .6, 1.]], dtype=float)
D = np.vectorize(prox2dist)(P)
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


def test_dc_single_source_vs_nx_single_source():
    """ Test Dijkstra: Rion's modified implementation vs Networkx original implementation > Single Source Dijkstra Path Length """
    # nx
    G = nx.from_edgelist(edgelist_james)
    nx.set_edge_attributes(G, name='weight', values=edgelist_james)
    nx_lengths = nx.single_source_dijkstra_path_length(G, source='s', weight='weight')

    # dc
    weight_function = _weight_function(G, 'weight')
    dc_lengths = single_source_dijkstra_path_length(G, source='s', weight_function=weight_function, disjunction=sum)

    assert (nx_lengths == dc_lengths)


def test_dc_cython_single_source_vs_nx_single_source():
    """ Test Dijkstra: Rion's implementation vs Networkx implementation > All Pairs """
    # dc
    G = nx.from_edgelist(edgelist_james)
    nx.set_edge_attributes(G, name='weight', values=edgelist_james)
    # cython
    nodes, edges, neighbors = from_networkx_to_dijkstra_format(G, weight='weight')
    cy_lengths = cy_single_source_dijkstra_path_length(source=0, N=nodes, E=edges, neighbors=neighbors, operator_names=('min', 'sum'))
    dc_lengths = {n: l for n, l in zip(G.nodes(), cy_lengths.values())}

    # nx
    nx_lengths = nx.single_source_dijkstra_path_length(G, source='s', weight='weight')

    assert (dc_lengths == nx_lengths)
