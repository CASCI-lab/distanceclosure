"""
import distanceclosure
from distanceclosure.utils import prox2dist
#
from distanceclosure.dijkstra import Dijkstra
from distanceclosure._dijkstra import _py_single_source_shortest_distances, _py_single_source_complete_paths
from distanceclosure.cython._dijkstra import _cy_single_source_shortest_distances, _cy_single_source_complete_paths
#
import networkx as nx
import random
from time import time

n = 1
G = nx.barabasi_albert_graph(n=n,m=2,seed=1)


P = nx.to_numpy_matrix(G).A
P[P!=0] = 0.5
D = prox2dist(P)
d = Dijkstra.from_numpy_matrix(D, directed=False)

print '--- Time it ---'

print '> Python _py_single_source_shortest_distances'
py_init_time = time()
for node in d.N:
	_py_single_source_shortest_distances(node, d.N, d.E, d.neighbours, (min,sum), verbose=10)
py_end_time = time()
py_time = py_end_time - py_init_time

#
print '> Cython _cy_single_source_shortest_distances'
cy_init_time = time()
for node in d.N:
	_cy_single_source_shortest_distances(node, d.N, d.E, d.neighbours, ('min','sum'), verbose=10)
cy_end_time = time()
cy_time = cy_end_time - cy_init_time

print '> Python'
py2_init_time = time()
d.all_pairs_shortest_distances(kind='metric', n_jobs=4, engine='python')
py2_end_time = time()
py2_time = py2_end_time - py2_init_time

print '> Cython'
cy2_init_time = time()
d.all_pairs_shortest_distances(kind='metric', engine='cython')
cy2_end_time = time()
cy2_time = cy2_end_time - cy2_init_time

print 'Python Dijkstra time: %s' % (py_time)
print 'Cython Dijkstra time: %s' % (cy_time)
print 'Python Parallel Dijkstra time: %s' % (py2_time)
print 'Cython Main time: %s' % (cy2_time)
"""
