# -*- coding: utf-8 -*-
"""
Transitive closure
===================

Computes transitive closure on a graph Adjacency Matrix.

These algorithms work with undirected weighted (distance) graphs.
"""
#    Copyright (C) 2015 by
#    Luis Rocha <rocha@indiana.edu>
#    Thiago Simas <@.>
#    Rion Brattig Correia <rionbr@gmail.com>
#    All rights reserved.
#    MIT license.
import numpy as np
import scipy.sparse as ssp
from dijkstra import Dijkstra
from itertools import izip
__name__ = 'distanceclosure'
__author__ = """\n""".join(['Luis Rocha <rocha@indiana.com>',
							'Thiago Simas <@.>',
							'Rion Brattig Correia <rionbr@gmail.com>'])
__all__ = ['transitive_closure']
__metrics__ = ['metric','ultrametric']
__algorithms__ = ['dense','dijkstra']
#
#
#
def transitive_closure(D, kind='metric', algorithm='dense', *args, **kwargs):
	"""
	Compute the transitive closure (All-Pairs-Shortest-Paths; APSP) using different shortest path measures
	on the distance graph (adjacency matrix) with values in the ``[0,inf]`` interval.
	
	.. math::
		c_{ij} = min_{k}( metric ( a_{ik} , b_{kj} ) )

	Args:
		D (matrix or dict): The [D]istance matrix. Accepted formats for kind ``dense`` is a numpy array; for ``dijkstra`` is a either a numpy array, a scipy sparse matrix or edgelist dictionary.
		kind (string): type of closure to compute: ``metric`` or ``ultrametric``.
		algorithm (string): type of algorithm to use: ``dense`` or ``dijkstra``.
		verbose (bool): Prints statements as it computes.

	Returns:
		C (matrix or dict): transitive closure dense matrix or a edgelist dictionary, depending on input
	
	Examples:

		>>> # using dense matrix
		>>> P = np.array([
			[1.,.9,.1,0.],
			[.9,1.,.8,0.],
			[.1,.8,1.,.6],
			[0.,0.,.6,1.],
			], dtype=float)
		>>> D = prox2dist(P)
		>>> transitive_closure(D, kind='metric', algorithm='dense', verbose=True)
			[ 0. ,.11111111, 0.36111111, 1.02777778],
			[ 0.11111111, 0., 0.25, 0.91666667],
			[ 0.36111111, 0.25, 0., 0.66666667],
			[ 1.02777778, 0.91666667, 0.66666667,  0.]

		>>> # using an edgelist
		>>> D = {
			('a','b'): 0.11111111,
			('a','c'): 9.,
			('b','c'): 0.25,
			('c','d'): 0.66666667,
		}
		>>> transitive_closure(D, kind='metric', algorithm='dijkstra', verbose=True)

		>>> # using a sparse matrix
		>>> Dsp = csr_matrix(D)
		>>> transitive_closure(Dsp, kind='metric', algorithm='dijkstra', verbose=True)
	
	Note:
		Dense matrix is slow for large graphs. If your network is large and/or sparse, use Dijkstra.

		Metric: :math:`(min,+)`

		Ultrametric: :math:`(min,max)` -- also known as maximum flow.
		
		Semantic proximity: TODO
		
		.. math::
			
				[ 1 + \\sum_{i=2}^{n-1} log k(v_i) ]^{-1}


	"""
	_check_for_metric_type(kind)
	_check_for_algorithm(algorithm)

	# Algorithm - Dense
	if algorithm == 'dense':
		
		# Numpy object
		if (type(D).__module__ == np.__name__):
			return _transitive_closure_dense_numpy(D, kind, *args, **kwargs)
		
		else:
			raise TypeError("Input is not a numpy object.")
	
	# Dijkstra
	elif algorithm == 'dijkstra':

		# Numpy object
		if (type(D).__module__ == np.__name__):
			dij = Dijkstra.from_numpy_matrix(D, *args, **kwargs)
		
		# Edgelist object
		elif (isinstance(D, dict)):
			dij = Dijkstra.from_edgelist(D, *args, **kwargs)

		# Sparse Matrix
		elif (ssp.issparse(D)):
			dij = Dijkstra.from_sparse_matrix(D, *args, **kwargs)

		else:
			raise TypeError("Invalid Input. For the Dijkstra algorithm, input must be `Numpy matrix`, `Scipy Sparse Matrix` or `Edgelist dict`")

		dij.all_pairs_shortest_paths(kind=kind)
		return dij.get_shortest_distances(format='sparse')
	

def _transitive_closure_dense_numpy(A, kind='metric', verbose=False):
	"""
	Calculates Transitive Closure using numpy dense matrix traversing.
	"""
	C = A.copy()
	n,m = A.shape
	
	# Check if diagonal is all zero
	if sum( np.diagonal(A) ) > 0:
		raise ValueError("Diagonal has to be zero for matrix computation to be correct")

	# Compute Transitive Closure
	for i in xrange(0,n):
		if verbose:
			print 'calc row:',i+1,'of',m
		for j in xrange(0,n):
			
			if kind == 'metric':
				vec = C[i,:] + C[:,j]
				C[i,j] = vec.min()

			elif kind == 'ultrametric':
				vec = np.maximum( C[i,:], C[:,j] )
				C[i,j] = vec.min()
	
	return np.array(C)
	

def _check_for_metric_type(kind):
	"""
	Check for available metric functions.
	"""
	if kind not in __metrics__:
		raise TypeError("Metric not found for this algorithm. Try 'metric' or 'ultrametric',")

def _check_for_algorithm(algorithm):
	"""
	Check for available algorithm.
	"""
	if algorithm not in __algorithms__:
		raise TypeError("Algorithm implementation not supported. Try 'dense', 'dijkstra' or leave blank.")


def S_measure(D, Cm):
	"""
	Computes the S measure for each network edge.
	The S measure is the ratio between the direct distance (from the original graph) and the indirect distance (from the metric closure graph).
	The formal definition is as follow:
	
	.. math::
		s_{ij} = d_{ij} / d_{ij}^m.

	Args:
		D (matrix): The [D]istance matrix.
		Cm (matrix): The Metric [C]losure matrix.
	
	Note: 
		Both arguments must be numpy arrays as the Metric Closure network is a dense matrix.
	"""

	# Assert both are numpy arrays
	if (type(D).__module__ != np.__name__) or (type(Cm).__module__ != np.__name__):
		raise TypeError("Both D and Cm must be Numpy objects.")

	# Item-wise division
	S = D / Cm
	# For edges where the distance is infite, set it to NAN
	S[D==np.inf] = np.nan

	return S
	

def B_measure(D, Cm, verbose=False):
	"""
	Computes the B measure for each network edge with infinite distance, thus not existing in the original Distance graph.
	The formal definition is as follow:
	
	.. math::
		b_{ij} = <d_{ik}> / d_{ij}^m
		b_{ji} = <d_{jk}> / d_{ij}^m

	which is the average distance of all edges that leaves from node `x_i`, divided by its new closure distance.
	Also note that `b_{ij}` can be different from `b_{ji}`.

	Args:
		D (matrix): The [D]istance adjacency matrix.
		Cm (matrix): The Metric [C]losure adjacency matrix.

	Note: 
		Both arguments must be numpy arrays as the Metric Closure network is a dense matrix.

	Warning:
		This computation takes a while.
	"""
	D = D.copy()

	B = np.empty(shape=D.shape)
	
	# Get edges that are INF
	rows, cols = np.where(D==np.inf)

	# Transform INF to NAN so we can ignore it on averaging
	D[D==np.inf] = np.nan
	means = np.nanmean(D, axis=1)

	for i, (row, col) in enumerate(izip(rows,cols)):
		B[row,col] = means[row] / float( Cm[row,col] )
		
		if verbose:
			if i%10000==0 and i>0:
				print i,'of',len(rows)

	return B

