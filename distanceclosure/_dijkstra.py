# -*- coding: utf-8 -*-
"""
Transitive closure
===================

Computes transitive closure on a graph Adjacency Matrix.

These algorithms work with undirected weighted (distance) graphs.
"""
#    Copyright (C) 2017 by
#    Rion Brattig Correia <rionbr@gmail.com>
#    All rights reserved.
#    MIT license.
import numpy as np
import heapq
__author__ = """\n""".join(['Rion Brattig Correia <rionbr@gmail.com>'])
#
#
#
def _py_single_source_shortest_distances(source, N, E, neighbors, operators=(min,sum), verbose=0):
	"""
	Compute shortest distance between a source and all other reachable nodes.

	Args:
		source (int): the source node.
		N (list): the list of nodes in the network.
		E (dict): the dict of edges in the network.
		neighbors (dict): a dict that contains all node neighbors.
		operators (tuple): a tuple of the operators to compute shortest path. Default is ``(min,max)``.
		verbose (int, optional): print statements as it computed shortest distances.

	Returns:
		dists (dict): the final distance calculated from source to all other nodes.
		paths (dict): the local path between source and all other nodes.

	Note:
		The python `heapq` module does not support item update.
		Therefore this algorithm keeps track of which nodes and edges have been searched already;
		and the queue itself has duplicated nodes inside.
	"""
	Q = [] # priority queue; note items are mutable
	final_dist = {} # {node:distance}
	paths = {} # {node: [distance, parent_node]}
	visited_nodes = set([]) # We need this because we can't update the heapq
	visited_edges = set([])
	disjf, conjf = operators

	for node in N:
		# Root node has distance 0
		if node == source:
			final_dist[source] = 0.
			heapq.heappush(Q, [0, node])
		# All other nodes have distance infinity
		else:
			final_dist[node] = np.inf
			heapq.heappush(Q, [np.inf, node])
	
	# Iterate over all nodes in the Queue
	while Q:
		node_dist, node = heapq.heappop(Q) # Current `node distance` and `node index`

		#If this node has been searched, continue
		if node in visited_nodes:
			continue
		
		# Iterate over all neighbors of node 
		for neighbor in neighbors[node]:
			
			# If this edge has been searched, continue
			if (node, neighbor) in visited_edges:
				continue

			# the edge distance/weight/cost
			weight = E[ (node, neighbor) ]
			
			# Operation to decide how to compute the lenght, summing edges (metric) or taking the max (ultrametric)
			new_dist = conjf([node_dist, weight])

			# If this is a shortest distance, update
			if new_dist < final_dist[neighbor]:
				# update the shortest distance to this node
				final_dist[neighbor] = new_dist
				# update (actually include a new one) this neighbor on the queue
				heapq.heappush(Q, [new_dist, neighbor])
				# update the path
				paths[neighbor] = [new_dist, node]

			# Add to visited edges
			visited_edges.add( (neighbor,node) )
		
		# Add to visited nodes
		visited_nodes.add(node)

	return final_dist, paths

def _py_single_source_complete_paths(source, N, paths):
	"""
	From the dict of node parent paths, recursively return the complete path from the source to all targets
	
	Args:
		source (int/string): the source node.
		N (dict): the set of nodes in the network.
		paths (dict): a dict of nodes and their distance to the parent node.
	
	Returns:
		path (dict): the complete path between source and all other nodes, including source and target in the list.
	
	"""
	def __py__get_path_recursive(plist, n, source):
		if n != source:
			plist.append(n)
			try:
				n = paths[n][1]
			except:
				pass
			else:
				__py__get_path_recursive(plist, n, source)
		return plist

	complete_paths = {}
	for n in N:
		plist = __py__get_path_recursive([], n, source)
		plist.append(source)
		plist.reverse()
		complete_paths[n] = plist

	return complete_paths

