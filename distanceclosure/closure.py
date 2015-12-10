# -*- coding: utf-8 -*-
"""
Compute the metric, ultrametric and semantic distance closure on the graph.

These algorithms work with undirected weighted (distance) graphs.

"""
#    Copyright (C) 2015 by
#    Luis Rocha <rocha@indiana.edu>
#    Thiago Simas <@.>
#    Rion Brattig Correia <rionbr@gmail.com>
#    All rights reserved.
#    MIT license.
import numpy as np

__author__ = """\n""".join(['Luis Rocha <rocha@indiana.com>',
							'Thiago Simas <@.>',
							'Rion Brattig Correia <rionbr@gmail.com>'])
__all__ = [
			'transitive_closure',
			'backbone_numpy',
		]

__metrics__ = ['metric','ultrametric']

def transitive_closure(A, kind='metric', *args, **kwargs):
	"""
	Compute the transitive closure (APSP) using different shortest path measures
	on the Distance Network Adjacency Matrix with values in the [0,inf] interval.
	
	- Metric: (min,+)
	- Ultrametric: (min,max)
	TODO:
	- Semantic Proximity: [ 1 + \sum_{i=2}^{n-1} log k(v_i) ]^{-1}
	
	Note: this algorithm uses numpy dense matrix calculations. Which is some slow.

		c_{ij} = min_{k}( metric ( a_{ik} , b_{kj} ) )

	Parameters
	----------
	M : Numpy matrix

    kind : string
    	type of closure to compute: 'metric', 'ultrametric' or 'semantic'

	Returns:
	------
	C : Numpy matrix
		the transitive closure graph
	
	Examples
    --------
	>>> C = transitive_closure_numpy(M, kind='metric')
	"""
	_check_for_metric_type(kind)
	
	# Numpy object
	if (type(A).__module__ == np.__name__):
		return _transitive_closure_numpy(A, kind, *args, **kwargs)
	else:
		raise TypeError("Input is not a numpy object")

def _transitive_closure_numpy(A, kind='metric', verbose=False):
	C = A.copy()
	n,m = A.shape

	# Check number of zeros in the matrix
	if len(A[A==0]) != n:
		raise ValueError("Only the diagonal has to be zero. All other zeros need to be 'numpy.inf'")
	
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


def _transitive_closure_dijkstra_matrix_all_pairs(A, kind):
	# TODO	
	return None
	n,m = A.shape

	for source in xrange(0,n):
		_transitive_closure_dijkstra_matrix(A, source, kind)

def _transitive_closure_dijkstra_matrix(A, source, kind):
	# TODO
	pass

#
# Backbone
#
def backbone(A, C):
	"""
	Return backbone edges based on the the original graph and its transitive closure.
	Values:
		1 = metric
		2 = semi-metric
		-1 = diagonal
		0 = otherwise

	Parameters
	----------
	A : dense or sparse array
		Adjacency matrix from original graph
	
	C : dense or sparse array
		Adjacency matrix from transitive closure graph

	Returns:
	------
	B : array
		Adjacency matrix where backbone edges are 1, semi-metric edges are 2, diagonal is -1, 0 otherwise.

	Examples
    --------
	>>> C = transitive_closure_numpy(A, kind='metric')
	>>> B = backbone_numpy(A, C)
	>>> rows, cols = np.where(B==1)
	"""	
	# Check for data type
	if (type(A).__module__ == np.__name__) and (type(C).__module__ == np.__name__):
		return _backbone_numpy(A, C)
	else:
		raise TypeError("Inputs are not valid objects, try numpy or scipy objects. Objects must be both the same type.")

def _backbone_numpy(A, C):
	n,m = A.shape
	B = np.zeros(A.shape)
	if n==m:
		np.fill_diagonal(B, -1)
	# Semi-metric values = 2
	rows, cols = np.where( (C > 0) & (C != np.inf) )
	B[rows,cols] = 2
	# Metric Values = 1
	rows, cols = np.where( (C > 0) & (C != np.inf) & np.isclose(A , C) )
	B[rows,cols] = 1
	return B


def _check_for_metric_type(kind):
	if kind not in __metrics__:
		raise TypeError("Metric not found for this algorithm. Try 'metric' or 'ultrametric',")

