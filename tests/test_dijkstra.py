from distanceclosure.dijkstra import Dijkstra
from distanceclosure.utils import prox2dist
import numpy as np
import networkx as nx
#
# Test
#
# Numpy
P = np.array([
		[1.,.9,.1,0.],
		[.9,1.,.8,0.],
		[.1,.8,1.,.6],
		[0.,0.,.6,1.],
		], dtype=float)
D = prox2dist(P)
#
# Edgelist
#
edgelist_luis = {
	('s','b'):.9,
	('s','c'):.1,
	('b','c'):.8,
	('c','d'):.6,
}

edgelist_james = {
	('s','a'):8,
	('s','c'):6,
	('s','d'):5,
	('a','d'):2,
	('a','e'):1,
	('b','e'):6,
	('c','d'):3,
	('c','f'):9,
	('d','f'):4,
	('e','g'):4,
	('f','g'):0,
}

# Single Source Shortest Paths > from Numpy
def test_dijkstra_sssp_from_numpy():
	""" Test Dijkstra: Single Source Shortest Path (SSSP) from Numpy """
	dij = Dijkstra.from_numpy_matrix(D, verbose=True)
	dij.single_source_shortest_paths(source=0, kind='metric')

# Single Source Shortest Paths > from edgelist
def test_dijkstra_sssp_from_edgelist():
	""" Test Dijkstra: Single Source Shortest Path (SSSP) from Edgelist """
	dij = Dijkstra.from_edgelist(edgelist_james, verbose=True)
	dij.single_source_shortest_paths(source='s', kind='metric')


def test_dijkstra_vs_networkx_single_source_all_lenghts_and_paths():
	""" Test Dijkstra: Rion's implementation vs Networkx implementation > Single Source """
	# NX Version
	G = nx.from_edgelist(edgelist_james) 
	nx.set_edge_attributes(G, 'weight', edgelist_james)
	nx_lenghts = nx.single_source_dijkstra_path_length(G, source='s', weight='weight')
	nx_paths = nx.single_source_dijkstra_path(G, source='s', weight='weight')
	
	# My Version
	d = Dijkstra.from_edgelist(edgelist_james, directed=False)
	dc_lenghts, dc_paths = d.single_source_shortest_paths('s', kind='metric')

	assert (nx_lenghts == dc_lenghts)
	assert (nx_paths == dc_paths)


def test_dijkstra_vs_networkx_apsp():
	""" Test Dijkstra: Rion's implementation vs Networkx implementation > All Pairs """
	# NX Version
	G = nx.from_edgelist(edgelist_james) 
	nx.set_edge_attributes(G, 'weight', edgelist_james)
	nx_all_complete_paths = nx.all_pairs_dijkstra_path(G, 'weight')

	# My Version
	d = Dijkstra.from_edgelist(edgelist_james, directed=False)
	dc_all_lenghts, dc_all_paths = d.all_pairs_shortest_paths()
	dc_all_complete_paths = d.shortest_complete_paths

	print d
	print d.N
	print d.E
	print 'nx_all_complete_paths'
	print nx_all_complete_paths
	print 'dc_all_paths'
	print dc_all_paths
	assert (nx_all_complete_paths == dc_all_complete_paths)


