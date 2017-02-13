# -*- coding: utf-8 -*-
"""
Dijkstra algorithm
===================

Implementation of a generalized version of the Dijkstra algorithm using Heap Queue and multiprocessing to compute transitive closure.
This algorithm uses a modification to the path length to compute both `metric` and `ultrametric` closure.

Warning:
	There is no need for this class to be called directly, since it will be called from :meth:`~distanceclosure.closure`.

Note:
	A very good tutorial of how the Djikstra algorithm works can be seen `here
	<https://www.youtube.com/watch?v=U9Raj6rAqqs>`_.

"""
#    Copyright (C) 2015 by
#    Rion Brattig Correia <rionbr@gmail.com>
#    Luis Rocha <rocha@indiana.edu>
#    Thiago Simas <@.>
#    All rights reserved.
#    MIT license.
from utils import dict2sparse
import numpy as np
import heapq
from joblib import Parallel, delayed
#import dill
from _dijkstra import _single_source_shortest_distances, _single_source_complete_paths


__name__ = 'distanceclosure'
__author__ = """\n""".join(['Rion Brattig Correia <rionbr@gmail.com>',
							'Luis Rocha <rocha@indiana.com>',
							'Thiago Simas <@.>'])
__all__ = ['Dijkstra']
__operators__ = {
	'metric': (min,sum),
	'ultrametric':(min,max)
	}



class Dijkstra(object):
	"""
	This is the class that handles the computation of the Distance Closure using a generalization of the Djikstra Algorithm.
	"""
	def __init__(self, N=set(), E=dict(), neighbours=dict(), directed=False, verbose=False, verbose_level=1):
		self.N = N 						# nodes
		self.E = E 						# edges
		self.neighbours = neighbours	# dict of neighbouring edges for every node
		self.directed = directed		# is graph directed?
		#
		self.verbose = verbose
		self.verbose_level = verbose_level
		#
		self.shortest_distances = {k:None for k in N}	# will be populated by `all_pairs_shortest_paths` or `single_source_shortest_paths`
		self.shortest_complete_paths = {k:None for k in N}

	def __str__(self):
		return "<Dijkstra Format Network(n_nodes=%d n_edges=%d, directed=%s, verbose=%s)>" % ( len(self.N) , len(self.E) , self.directed , self.verbose )

	def all_pairs_shortest_paths(self, kind='metric', n_jobs=1):
		"""
		Computes All Pairs Shortest Paths (APSP).

		Args:
			kind (string): The metric type. 'metric' or 'ultrametric' are currently accepted.
			n_jobs (int): The number of CPUs to use to do the computation. ``-1`` means 'all CPUs'.

		Returns:
			shortest_distances (dict): A dict-of-dicts of distances between all pair of nodes. Ex: ``{'a':{'c':0.1}}``
			shortest_complete_paths (dict): A dict-of-dicts-of-list of the shortest path between all pair of nodes. Ex: ``{'a':{'c':['a','b','c']}}``

		Note:
			The same as calling :func:`single_source_shortest_paths` for every node as a source.

		"""
		if self.verbose:
			print 'Calculating APSP - All Pairs Shortest Paths/Distances'

		try:
			operators = __operators__[kind]
		except Exception, e:
			raise AttributeError("kind parameter must be either 'metric' or 'ultrametric'")

		# Shortest Distances in Parallel
		poolresult = Parallel(n_jobs=n_jobs)(delayed(_single_source_shortest_distances)(node, self.N, self.E, self.neighbours, operators) for node in self.N)

		# PoolResutls returns a tuple, separate the two variables
		shortest_distances, shortest_complete_paths = map(list, zip(*poolresult))

		# Then turn them into dict-of-dicts
		self.shortest_distances = dict(zip(self.N, shortest_distances))
		self.shortest_complete_paths = dict(zip(self.N, shortest_complete_paths))

		return self.shortest_distances, self.shortest_complete_paths

	def single_source_shortest_paths(self, source, kind='metric'):
		"""
		Computes Single Source Shortest Path (SSSP)

		Args:
			source (int or string): the source node to compute shortest paths to every other node.
			kind (string): The metric type. 'metric' or 'ultrametric' are currently accepted.

		Returns:
			shortest_distances (dict): A dict of distances between the source all other nodes.
			shortest_complete_paths (dict): A dict-of-list of the shortest distance path between the source and all other nodes. Ex: ``{'c':['a','b','c']}}``
		"""
		try:
			operators = __operators__[kind]
		except Exception, e:
			raise AttributeError("kind parameter must be either 'metric' or 'ultrametric'")

		# Shortest Distances
		shortest_distance, shortest_complete_path = _single_source_shortest_distances(source, self.N, self.E, self.neighbours, operators)
		print shortest_distance
		# Save to object
		self.shortest_distances[source] = shortest_distance
		self.shortest_complete_paths[source] = shortest_complete_path

		return shortest_distance, shortest_complete_path

	@classmethod
	def from_edgelist(self, edgelist, directed=False, *args, **kwargs):
		"""
		Instanciantes the class from a edgelist dictionary.

		Args:
			edgelist (dict): Distance graph edgelist adjacency matrix.

		Examples:
			>>> edgelist_luis = {
				('s','b'):.9,
				('s','c'):.1,
				('b','c'):.8,
				('c','d'):.6,
			}
			>>> dij = Dijkstra(verbose=True)
		"""
		# Init
		N = set()
		E = dict()
		neighbours = dict()

		# Create dictionaries to be used to compute single-source-shortest-paths
		for (v1,v2),d in edgelist.items():
			if v1 not in N:
				N.add(v1)
				neighbours[v1] = []
			if v2 not in N:
				N.add(v2)
				neighbours[v2] = []
			
			# If indirected graph, include both directions
			if not directed:
				E[ (v2,v1) ] = d
				neighbours[v2].append(v1)
			E[ (v1,v2) ] = d
			neighbours[v1].append(v2)
		
		return Dijkstra(N, E, neighbours, directed, *args, **kwargs)

	@classmethod
	def from_numpy_matrix(self, matrix, directed=False, *args, **kwargs):
		"""
		Instanciantes the class from a Numpy adjacency matrix.

		Args:
			matrix (matrix): Distance graph Numpy adjacency matrix.

		Examples:
			>>> P = np.array([
					[1.,.9,.1,0.],
					[.9,1.,.8,0.],
					[.1,.8,1.,.6],
					[0.,0.,.6,1.],
					], dtype=float)
			>>> D = prox2dist(P)
			>>> dij = Dijkstra(verbose=True)
			>>> dij.from_numpy_matrix(D)
		"""
		# Init
		N = set()
		E = dict()
		neighbours = dict()

		# Assert Square Adjacency Matrix
		if matrix.shape[0] != matrix.shape[1]:
			raise ValueError('Adjacency Matrix not square')

		N = set( np.arange(matrix.shape[0]) )
		for i, row in enumerate(matrix,start=0):
			neighbours[i] = []
			for j, value in enumerate(row,start=0):				
				# the diagonal is (must be) always zero (distance = 0)
				if i==j:
					continue
				# infinite distance doesn't have to be calculated
				elif value == np.inf:
					continue
				else:
					E[ (i,j) ] = float(value)
					neighbours[i].append(j)

		return Dijkstra(N, E, neighbours, directed, *args, **kwargs)
	
	@classmethod
	def from_sparse_matrix(self, matrix, directed=False, *args, **kwargs):
		"""
		Instanciantes the algorithm from a Scipy sparse adjacency matrix.

		Args:
			matrix (sparse matrix) : Distance graph Scipy sparse adjacency matrix.

		Examples:
			>>> Dsp = csr_matrix(D)
			>>> dij = Dijkstra(verbose=True)
			>>> dij.from_sparse_matrix(Dsp)
		"""
		# Init
		N = set()
		E = dict()
		neighbours = dict()

		# Assert Square Adjacency Matrix
		if matrix.shape[0] != matrix.shape[1]:
			raise ValueError('Adjacency Matrix not square')

		N = set( np.arange(matrix.shape[0]) )
		neighbours = [[] for _ in np.arange(matrix.shape[0])]
		#
		rows,cols = matrix.nonzero()
		for i,j in zip(rows,cols):
			# the diagonal is (must be) always zero (distance = 0)
			if i==j:
				continue
			# infinite distance doesn't have to be calculated
			elif matrix[i,j] == np.inf:
				continue
			else:
				E[ (i,j) ] = float(matrix[i,j])
				neighbours[i].append(j)

		return Dijkstra(N, E, neighbours, directed, *args, **kwargs)

	def get_shortest_distances(self, format='sparse'):
		"""
		After the computation of APSP, returns the shortest distances in Scipy sparse format.

		Returns:
			M (sparse matrix) : 
		"""
		if format == 'sparse':
			return dict2sparse(self.shortest_distances)
		else:
			return self.shortest_distances

if __name__ == '__main__':

	import networkx as nx
	from distanceclosure.utils import prox2dist
	from scipy.sparse import csr_matrix

	# edge list
	#https://www.youtube.com/watch?v=U9Raj6rAqqs
	edgelist = {
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
	"""
	edgelist = {
		('s','b'):.9,
		('s','c'):.1,
		('b','c'):.8,
		('c','d'):.6,
	}
	"""
	matrix = np.array([
			[1.,.9,.1,0.],
			[.9,1.,.8,0.],
			[.1,.8,1.,.6],
			[0.,0.,.6,1.],
			], dtype=float)
	matrix = prox2dist(matrix)

	sparse = csr_matrix(matrix)

	source = 2
	# NX
	#G = nx.from_edgelist(edgelist) 
	G = nx.from_numpy_matrix(matrix)
	#nx.set_edge_attributes(G, 'weight', edgelist)
	nx_lenghts = nx.single_source_dijkstra_path_length(G, source=source, weight='weight')
	nx_paths = nx.single_source_dijkstra_path(G, source=source, weight='weight')

	#d = Dijkstra(verbose=True)
	#d = Dijkstra.from_edgelist(edgelist, directed=False)
	d = Dijkstra.from_numpy_matrix(matrix, directed=False)
	#d = Dijkstra.from_sparse_matrix(sparse, directed=False, verbose=True)
	
	print '=== SSSP ==='
	print 'Source:',source
	print '---'
	dc_lenghts, dc_paths = d.single_source_shortest_paths(source=source, kind='metric')
	
	print '-- NX Results: --'
	print 'Lenghts:',nx_lenghts
	print 'Paths:',nx_paths

	print '--- DC Results ---'
	print 'Lenghts:',dc_lenghts
	print 'Paths:',dc_paths

	assert (nx_lenghts == dc_lenghts)
	assert (nx_paths == dc_paths)

	print '=== APSP ==='	

	nx_all_complete_paths = nx.all_pairs_dijkstra_path(G, 'weight')

	dc_all_lenghts, dc_all_complete_paths = d.all_pairs_shortest_paths(n_jobs=2)
	
	print '-- NX Results: --'
	print 'Paths:',nx_all_complete_paths

	print '-- DC Results: --'
	print 'Lenghts;',dc_all_lenghts
	print 'Paths:',dc_all_complete_paths

	print '==='

	print nx_all_complete_paths[2]
	print dc_all_complete_paths[2]

	assert (nx_all_complete_paths == dc_all_complete_paths)

