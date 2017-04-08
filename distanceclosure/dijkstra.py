# -*- coding: utf-8 -*-
"""
Dijkstra algorithm
===================

Implementation of a generalized version of the Dijkstra algorithm using Heap Queue and multiprocessing to compute transitive closure.
This algorithm uses a modification to the path length to compute both `metric` and `ultrametric` closure.

Warning:
	There is no need for this class to be called directly, since it will be called from :meth:`~distanceclosure.closure`.
	This algoritm currentrly only works on undirected networks.

Note:
	A very good introductory tutorial of how the Djikstra algorithm works can be seen `here
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
from _dijkstra import _py_single_source_shortest_distances, _py_single_source_complete_paths
from cython._dijkstra import _cy_single_source_shortest_distances, _cy_single_source_complete_paths


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
	Under the hood it has two implementations, in Cython and Python, both using a priority heap queue.
	"""
	def __init__(self, N=list(), E=dict(), neighbours=dict(), node_names=list(), directed=False, verbose=0):
		self.N = N 						# nodes
		self.E = E 						# edges
		self.node_names = node_names 	# node names

		self.neighbours = neighbours	# dict of neighbouring edges for every node
		self.directed = directed		# is graph directed?
		#
		self.verbose = verbose
		#
		self.shortest_distances = {k:None for k in N}	# will be populated by `all_pairs_shortest_paths` or `single_source_shortest_paths`
		self.local_paths = {k:None for k in N}
		self.shortest_paths = {k:None for k in N}

	def __str__(self):
		return "<Dijkstra Format Network(n_nodes=%d n_edges=%d, directed=%s, verbose=%s)>" % ( len(self.N) , len(self.E) , self.directed , self.verbose )

	def all_pairs_shortest_distances(self, kind='metric', n_jobs=1, engine='cython', verbose=0):
		"""
		Computes All Pairs Shortest Distances (APSD).

		Args:
			kind (string): The metric type. 'metric' or 'ultrametric' are currently accepted.
			n_jobs (int, optional): The number of CPUs to use to do the computation. ``-1`` means 'all CPUs'.
				Only available for 'python' engine.
			engine (string): The implementation to use. Either ``cython`` or ``python``.
			verbose (int, optional): The verbosity level: if non zero, progress messages are printed.
				Above 50, the output is sent to stdout. The frequency of the messages increases with the verbosity level.
				If it more than 10, all iterations are reported.

		Returns:
			shortest_distances (dict): A dict-of-dicts of distances between all pair of nodes.
				Ex: ``{'a':{'c':0.1,'b':0.3}, ...}``.
			local_paths (dict): A dict-of-dicts-of-list of the shortest local path between all pair of nodes.
				Ex: ``{'a': {'b': [0.1, 'a'], 'c': [0.3, 'b']} , ...}``.

		Note:
			The same as calling :func:`single_source_shortest_distances` for every node in `Parallel`.

		"""
		if self.verbose:
			print 'Calculating APSD - All Pairs Shortest Distances'

		try:
			operators = __operators__[kind]
		except Exception, e:
			raise AttributeError("kind parameter must be either 'metric' or 'ultrametric'")

		# Shortest Distances in Parallel
		if engine == 'python':
			
			poolresults = Parallel(n_jobs=n_jobs,verbose=verbose)(delayed(_py_single_source_shortest_distances)(node, self.N, self.E, self.neighbours, operators, verbose) for node in self.N)
		
		elif engine == 'cython':
			# cython uses its own sum and max functions. So let's just pass their names.
			operators = (operators[0].__name__ , operators[1].__name__)
			#
			poolresults = range(len(self.N))
			for node in self.N:
				poolresults[node] = _cy_single_source_shortest_distances(node, self.N, self.E, self.neighbours, operators, verbose)

		# PoolResults returns a tuple, separate the two variables
		shortest_distances, local_paths = map(list, zip(*poolresults))

		# Then turn them into dict-of-dicts
		self.shortest_distances = dict(zip(self.N, shortest_distances))
		self.local_paths = dict(zip(self.N, local_paths))

		return self.shortest_distances, self.local_paths

	def all_pairs_shortest_paths(self, n_jobs=1, engine='cython', verbose=0, *args, **kwargs):
		"""
		Computes All Pair Shortest Paths (APSP)

		Args:
			n_jobs (int, optional): The number of CPUs to use to do the computation. ``-1`` means 'all CPUs'.
			engine (string): The implementation to use. Either ``cython`` or ``python``.
			verbose (int, optional): The verbosity level: if non zero, progress messages are printed. Above 50, the output is sent to stdout.
				The frequency of the messages increases with the verbosity level. If it more than 10, all iterations are reported.
		
		Returns:
			shortest_paths (dict): A dict-of-dicts-of-list of the shortest path between all pair of nodes.
				Ex: ``{'a':{'c':['a','b','c']}}``.
		"""
		if self.verbose:
			print 'Calculating APSP - All Pairs Shortest Paths'

		for path in self.local_paths:
			if path is None:
				raise Exception("Shortest distances and local paths must be calculated first. Run `all_pairs_shortest_distances`.")

		if engine == 'python':
			poolresults = Parallel(n_jobs=n_jobs,verbose=verbose)(delayed(_py_single_source_complete_paths)(node, self.N, self.local_paths[node]) for node in self.N)
		elif engine == 'cython':
			#
			poolresults = range(len(self.N))
			for node in self.N:
				poolresults[node] = _cy_single_source_complete_paths(node, self.N, self.local_paths[node])

		# PoolResults returns a list, map into a dict of nodes
		self.shortest_paths = dict( zip( self.N , poolresults ) )

		return self.shortest_paths

	def single_source_shortest_distances(self, source, kind='metric', engine='cython'):
		"""
		Computes Single Source Shortest Distances (SSSD)

		Args:
			source (int or string): the source node to compute shortest distances to every other node.
			kind (string): The metric type. 'metric' or 'ultrametric' are currently accepted.
			engine (string): The implementation to use. Either ``cython`` or ``python``.

		Returns:
			shortest_distances (dict): A dict of distances between the source all other nodes.
				Ex: ``{'c':0.1,'b':0.3}``.
			shortest_paths (dict): A dict-of-list of the shortest distance path between the source and all other nodes.
				Ex: ``{'b': [0.1, 'a'], 'c': [0.3, 'b']}``.
		"""
		if not isinstance(source, int):
			source = self.node_names.index(source)
		
		try:
			operators = __operators__[kind]
		except Exception, e:
			raise AttributeError("kind parameter must be either 'metric' or 'ultrametric'")

		# Shortest Distances
		if engine == 'python':
			shortest_distances, local_paths = _py_single_source_shortest_distances(source, self.N, self.E, self.neighbours, operators)
		elif engine == 'cython':
			operators = (operators[0].__name__ , operators[1].__name__)
			shortest_distances, local_paths = _cy_single_source_shortest_distances(source, self.N, self.E, self.neighbours, operators)

		# Save to object
		self.shortest_distances[source] = shortest_distances
		self.local_paths[source] = local_paths

		return shortest_distances, local_paths

	def single_source_shortest_paths(self, source, engine='cython', *args, **kwargs):
		"""
		Computes Single Source Shortest Paths (SSSP)

		Args:
			source (int or string): the source node to compute the shortest paths to every other node.
			engine (string): The implementation to use. Either ``cython`` or ``python``.

		Returns:
			shortest_paths (dict): A dict-of-list of the shortest distance path between the source and all other nodes. Ex: ``{'c':['a','b','c']}}``
		"""
		if not isinstance(source, int):
			source = self.node_names.index(source)

		if self.local_paths[source] is None:
			# Calculates the local paths in case it hasn't been calculated.
			raise Exception ("Shortest distances and local paths must be calculated first. Run `single_source_shortest_distances` or `all_pairs_shortest_distances`.")
		
		# Shortest Paths
		if engine == 'python':
			shortest_paths = _py_single_source_complete_paths(source, self.N, self.local_paths[source])
		elif engine == 'cython':
			shortest_paths = _cy_single_source_complete_paths(source, self.N, self.local_paths[source])

		# Save to object
		self.shortest_paths[source] = shortest_paths

		return shortest_paths

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
			>>> dij = Dijkstra.from_edgelist(edgelist_luis,verbose=True)
		"""
		N = list()
		E = dict()
		v1i = None; v2i = None; i = 0
		node_names = list()
		neighbours = dict()

		# Create dictionaries to be used to compute single-source-shortest-paths
		for (v1,v2),d in edgelist.items():
			try:
				d = float(d)
			except:
				raise TypeError('Edge weights must numeric (int or float).')
			# Node 1
			if v1 not in node_names:
				v1i = i
				node_names.append(v1)
				neighbours[i] = list()
				i += 1
			else:
				v1i = node_names.index(v1)

			# Node 2
			if v2 not in node_names:
				v2i = i
				node_names.append(v2)
				neighbours[i] = list()
				i += 1
			else:
				v2i = node_names.index(v2)
			
			# Edges
			if not directed:
				# If indirected graph, include both directions
				E[ (v2i,v1i) ] = d
				neighbours[v2i].append(v1i)
			E[ (v1i,v2i) ] = d
			neighbours[v1i].append(v2i)
		
		N = range(len(node_names))

		return Dijkstra(N, E, neighbours, node_names, directed, *args, **kwargs)

	@classmethod
	def from_numpy_matrix(self, matrix, node_names=None, directed=False, *args, **kwargs):
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
			>>> dij = Dijkstra.from_numpy_matrix(D)
		"""
		N = list()
		E = dict()
		neighbours = dict()

		# Assert Square Adjacency Matrix
		if matrix.shape[0] != matrix.shape[1]:
			raise ValueError('Adjacency Matrix not square')

		#matrix = matrix.A

		N = list( np.arange(matrix.shape[0]) )
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

		return Dijkstra(N, E, neighbours, node_names, directed, *args, **kwargs)
	
	@classmethod
	def from_sparse_matrix(self, matrix, node_names=None, directed=False, *args, **kwargs):
		"""
		Instanciantes the algorithm from a Scipy sparse adjacency matrix.

		Args:
			matrix (sparse matrix) : Distance graph Scipy sparse adjacency matrix.

		Examples:
			>>> Dsp = csr_matrix(D)
			>>> dij = Dijkstra.from_sparse_matrix(Dsp,verbose=True)
		"""
		N = list()
		E = dict()
		neighbours = dict()

		# Assert Square Adjacency Matrix
		if matrix.shape[0] != matrix.shape[1]:
			raise ValueError('Adjacency Matrix not square')

		N = list( np.arange(matrix.shape[0]) )
		neighbours = {i:[] for i in np.arange(matrix.shape[0])}
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

		return Dijkstra(N, E, neighbours, node_names, directed, *args, **kwargs)


	def get_shortest_distances(self, source=None, translate=False, format='dict'):
		"""
		After the computation of APSD, returns the shortest distances.

		Args:
			source (int/name, optional): Return distances only from a specific source.
			translate (bool, optional): Translate node indices into the specified node names.
				This translation can ge quite expensive.
			format (string, optional): The returning format. Default is ``dict``.

		Returns:
			M (dict/matrix) : Returns the format specified on ``format`` arg.
		"""
		if source is not None:
			if not isinstance(source, int):
				source = self.node_names.index(source)
			sd = self.shortest_distances[source]
		else:
			sd = self.shortest_distances

		# Translate indices into node names
		if (self.node_names is not None) and (translate == True):
			sd = self._translate_indices_to_node_names(sd, self.node_names)

		# Format Conversion
		if format == 'sparse':
			return dict2sparse(sd)
		else:
			return sd

	def get_shortest_paths(self, source=None, translate=False):
		"""
		After the computation of APSP, returns the shortest distances.

		Args:
			source (int/name, optional): Return paths only from a specific source.
			translate (bool, optional): Translate node indices into the specified node names.
				This translation can ge quite expensive.

		Returns:
			M (dict/matrix) : Returns the format specified on ``format`` arg.
		"""
		if source is not None:
			if not isinstance(source,int):
				source = self.node_names.index(source)

			sp = self.shortest_paths[source]
		else:
			sp = self.shortest_paths

		# Translate indices into node names
		if (self.node_names is not None) and (translate == True):
			sp = self._translate_indices_to_node_names(sp, self.node_names)

		return sp

	def _translate_indices_to_node_names(self, d, names):
		"""
		Translates a dict-of-dict, from keys of numeric indices to keys of name strings.
		
		Args:
			d (dict): a dict of dicts.
			names (list): a list of strings with the names to be translated.

		Returns:
			d (dict): a translated dict of dicts
		"""	

		def __translate(obj, names):
			""" Recursive translate indices into node names """
			if isinstance(obj, int):
				return names[obj]
			elif isinstance(obj, list):
				return [__translate(x, names) for x in obj]
			elif isinstance(obj, dict):
				new_obj = {}
				for k,v in obj.items():
					new_obj[__translate(k, names)] = __translate(v, names)
				return new_obj
			else:
				return obj

		new_dict = __translate(d, names)

		return new_dict


if __name__ == '__main__':

	import networkx as nx
	from distanceclosure.utils import prox2dist
	from scipy.sparse import csr_matrix

	# edge list
	#https://www.youtube.com/watch?v=U9Raj6rAqqs
	edgelist = {
		('s','a'):8.,
		('s','c'):6.,
		('s','d'):5.,
		('a','d'):2.,
		('a','e'):1.,
		('b','e'):6.,
		('c','d'):3.,
		('c','f'):9.,
		('d','f'):4.,
		('e','g'):4.,
		('f','g'):0.,
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

	source = 's'

	# NX
	G = nx.from_edgelist(edgelist)
	nx.set_edge_attributes(G, 'weight', edgelist)
	#G = nx.from_numpy_matrix(matrix)
	
	nx_lenghts = nx.single_source_dijkstra_path_length(G, source=source, weight='weight')
	nx_paths = nx.single_source_dijkstra_path(G, source=source, weight='weight')

	d = Dijkstra.from_edgelist(edgelist, directed=False, verbose=True)
	#d = Dijkstra.from_numpy_matrix(matrix, directed=False, verbose=True)
	#d = Dijkstra.from_sparse_matrix(sparse, directed=False, verbose=True)
	
	print '=== SSSP ==='
	print '> Source:',source
	print '---'
	dc_lenghts, dc_paths = d.single_source_shortest_distances(source=source, kind='metric', engine='python')
	dc_paths = d.single_source_shortest_paths(source=source, engine='python')
	
	print '-- NX Results: --'
	print '> Lenghts:',nx_lenghts
	print '> Paths:',nx_paths

	print '--- DC Results ---'
	print '> Lenghts:',d.get_shortest_distances(source=source, translate=True)
	print '> Paths:',d.get_shortest_paths(source=source, translate=True)

	assert (nx_lenghts == d.get_shortest_distances(source=source, translate=True))
	#assert (nx_paths == dc_paths)

	print '=== APSP ==='	

	nx_all_complete_paths = nx.all_pairs_dijkstra_path(G, 'weight')

	dc_all_lenghts, dc_all_local_paths = d.all_pairs_shortest_distances(n_jobs=2, engine='python')
	dc_all_complete_paths = d.all_pairs_shortest_paths(n_jobs=2, engine='python')
	
	print '-- NX Results: --'
	print '> Paths:',nx_all_complete_paths

	print '-- DC Results: --'
	print '> Lenghts;',dc_all_lenghts
	print '> Paths:',dc_all_complete_paths

	print '==='
	print nx_all_complete_paths['s']
	#print dc_all_complete_paths[0]
	print d.get_shortest_paths(translate=True)['s']

	assert (nx_all_complete_paths == d.get_shortest_paths(translate=True))



