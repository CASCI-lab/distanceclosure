# -*- coding: utf-8 -*-
"""
Dijkstra algorithm
===================

Implementation of the Dijkstra algorithm using Heap Queue to compute transitive closure.
This algorithm uses a modification to the path length to compute both `metric` and `ultrametric` closure.

Warning:
	There is no need for this class to be called directly, since it will be called from :meth:`~distanceclosure.closure`.

Note:
	A very good tutorial of how the Djikstra algorithm works can be seen `here
	<https://www.youtube.com/watch?v=U9Raj6rAqqs>`_.

"""
#    Copyright (C) 2015 by
#    Luis Rocha <rocha@indiana.edu>
#    Thiago Simas <@.>
#    Rion Brattig Correia <rionbr@gmail.com>
#    All rights reserved.
#    MIT license.
from utils import dict2sparse
import numpy as np
import heapq

__name__ = 'distanceclosure'
__author__ = """\n""".join(['Luis Rocha <rocha@indiana.com>',
							'Thiago Simas <@.>',
							'Rion Brattig Correia <rionbr@gmail.com>'])
__all__ = ['Dijkstra']
__operators__ = {
	'metric': (min,sum),
	'ultrametric':(min,max)
	}

class Dijkstra(object):
	"""
	This is the class that handles the computation of the Distance Closure using a generalization of the Djikstra Algorithm.
	"""
	def __init__(self, verbose=False):
		self.N = set() 	# nodes
		self.E = {} 	# edges
		self.neighbours = {} 	# neighbour dict for every node
		self.directed = None 	# is graph directed?
		self.verbose = verbose 
		#
		self.shortest_paths = {} 	# {source: {target:path}}
		self.shortest_complete_paths = {} 	# {source: {target:[n_1,n_2,...,n_i]}}
		self.shortest_distances = {} 	# {source: {target:length}} | also known as lenghts
		

	def all_pairs_shortest_paths(self, kind='metric'):
		"""
		Computes All Pairs Shortest Paths (APSP).

		Note:
			Practically, it calls :func:`single_source_shortest_paths` for every node as a source.

		"""
		if self.verbose:
			print 'Calculating APSP - All Pairs Shortest Paths/Distances'
		for node in self.N:
			if self.verbose:
				print 'node:',node
			_final_dist, _complete_paths = self.single_source_shortest_paths(source=node, kind=kind)

		return self.shortest_distances, self.shortest_paths

	def single_source_shortest_paths(self, source, kind='metric'):
		"""
		Compute shortest path between source and all other reachable nodes.

		Note:
			The python `heapq` module does not support item update.
			Therefore this algorithm keeps track of which nodes and edges have been searched already;
			and queue itself has duplicated nodes inside.
		"""
		Q = [] # priority queue; note items are mutable
		final_dist = {} # {node:distance}
		paths = {} # {node: [distance, parent_node]}
		visited_nodes = set([]) # We need this because we can't update the heapq
		visited_edges = set([])
		disjf, conjf = __operators__[kind]

		for node in self.N:
			# Root node has distance 0
			if node == source:
				final_dist[source] = 0
				heapq.heappush(Q, [0, node])
			# All other nodes have distance infinity
			else:
				final_dist[node] = np.inf
				heapq.heappush(Q, [np.inf, node])
		
		# Iterate over all nodes in the Queue
		while Q:
			node_dist, node = heapq.heappop(Q) # Curent `node distance` and `node index`

			#If this node has been searched, continue
			if node in visited_nodes:
				continue
			
			if self.verbose:
				print '-Node:',node, '| node distance:', node_dist

			# Iterate over all neighbours of node 
			for v in self.neighbours[node]:
				
				# If this edge has been searched, continue
				if (node, v) in visited_edges:
					continue

				# the edge distance/weight/cost
				weight = self.E[ (node, v) ]
				if self.verbose:
					print 'neihbour:',v , '| weight:', weight
				
				# Operation to decide how to compute the lenght, summing edges (metric) or taking the max (ultrametric)
				new_dist = conjf([node_dist, weight])

				# If this is a shortest distance, update
				if new_dist < final_dist[v]:
					# update the shortest distance to this node
					final_dist[v] = new_dist
					# update (actually include a new one) this node on the queue
					heapq.heappush(Q, [new_dist, v])
					# update the path
					paths[v] = [new_dist, node]

				visited_edges.add( (v,node) )

		# Update object
		self.shortest_distances[source] = final_dist
		self.shortest_paths[source] = paths

		# get complete paths
		complete_paths = self._compute_complete_paths(source, paths)

		return final_dist, complete_paths

	def _compute_complete_paths(self, source, pathlinks):
		"""
		From the dict of node parent, recursively return the complete path from the source to all targets
		"""
		def __get_path_recursive(plist, n, source):
			if n != source:
				plist.append(n)
				try:
					n = pathlinks[n][1]
				except:
					pass
				else:
					__get_path_recursive(plist, n, source)
			return plist

		complete_paths = {}
		for n in self.N:
			plist = __get_path_recursive([], n, source)
			plist.append(source)
			plist.reverse()
			complete_paths[n] = plist
		
		# update object
		self.shortest_complete_paths[source] = complete_paths

		return complete_paths

	def from_edgelist(self, edgelist, directed=False):
		"""
		Instanciantes the algorithm from a edgelist dictionary.

		Args:
			edgelist (dict) : Distance graph edgelist adjacency matrix.

		Examples:
			>>> edgelist_luis = {
				('s','b'):.9,
				('s','c'):.1,
				('b','c'):.8,
				('c','d'):.6,
			}
			>>> dij = Dijkstra(verbose=True)
		"""
		self.directed = directed
		# Create dictionaries to be used to compute single-source-shortest-paths
		for (v1,v2),d in edgelist.items():
			if v1 not in self.N:
				self.N.add(v1)
				self.neighbours[v1] = []
			if v2 not in self.N:
				self.N.add(v2)
				self.neighbours[v2] = []
			
			# If indirected graph, include both directions
			if not directed:
				self.E[ (v2,v1) ] = d
				self.neighbours[v2].append(v1)
			self.E[ (v1,v2) ] = d
			self.neighbours[v1].append(v2)

	def from_numpy_matrix(self, matrix, directed=False):
		"""
		Instanciantes the algorithm from a Numpy adjacency matrix.

		Args:
			matrix (matrix) : Distance graph Numpy adjacency matrix.

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
		# Assert Square Adjacency Matrix
		if matrix.shape[0] != matrix.shape[1]:
			raise ValueError('Adjacency Matrix not square')

		self.directed = directed
		self.N = set( np.arange(matrix.shape[0]) )
		for i, row in enumerate(matrix,start=0):
			self.neighbours[i] = []
			for j, value in enumerate(row,start=0):				
				# the diagonal is (must be) always zero (distance = 0)
				if i==j:
					continue
				# infinite distance doesn't have to be calculated
				elif value == np.inf:
					continue
				else:
					self.E[ (i,j) ] = float(value)
					self.neighbours[i].append(j)

	def from_sparse_matrix(self, matrix, directed=False):
		"""
		Instanciantes the algorithm from a Scipy sparse adjacency matrix.

		Args:
			matrix (sparse matrix) : Distance graph Scipy sparse adjacency matrix.

		Examples:
			>>> Dsp = csr_matrix(D)
			>>> dij = Dijkstra(verbose=True)
			>>> dij.from_sparse_matrix(Dsp)
		"""
		# Assert Square Adjacency Matrix
		if matrix.shape[0] != matrix.shape[1]:
			raise ValueError('Adjacency Matrix not square')

		self.directed = directed
		self.N = set( np.arange(matrix.shape[0]) )
		self.neighbours = [[] for _ in np.arange(matrix.shape[0])]
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
				self.E[ (i,j) ] = float(matrix[i,j])
				self.neighbours[i].append(j)

	def get_shortest_distances_sparse(self):
		"""
		After the computation of APSP, returns the shortest distances in Scipy sparse format.

		Returns:
			M (sparse matrix) : 
		"""
		return dict2sparse(self.shortest_distances)

if __name__=='__main__':

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

	d = Dijkstra(verbose=True)
	#d.from_edgelist(edgelist, directed=False)
	#d.from_numpy_matrix(matrix, directed=False)
	d.from_sparse_matrix(sparse, directed=False)
	
	print '=== SSSP ==='
	print 'Source:',
	dc_lenghts, dc_paths = d.single_source_shortest_paths(source, kind='metric')
	
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

	dc_all_lenghts, dc_all_paths = d.all_pairs_shortest_paths()
	dc_all_complete_paths = d.shortest_complete_paths
	
	print '-- NX Results: --'
	print nx_all_complete_paths

	print '-- DC Results: --'
	print dc_all_complete_paths

	print '==='

	print nx_all_complete_paths[2]
	print dc_all_complete_paths[2]

	assert (nx_all_complete_paths == dc_all_complete_paths)

