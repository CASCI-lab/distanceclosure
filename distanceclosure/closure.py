# -*- coding: utf-8 -*-
"""
Transitive Closure
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
__name__ = 'distanceclosure'
__author__ = """\n""".join(['Luis Rocha <rocha@indiana.com>',
							'Thiago Simas <@.>',
							'Rion Brattig Correia <rionbr@gmail.com>'])
__all__ = ['transitive_closure']
__metrics__ = ['metric','ultrametric']
#
#
#
def transitive_closure(D, kind='metric', *args, **kwargs):
	"""
	Compute the transitive closure (All-Pairs-Shortest-Paths; APSP) using different shortest path measures
	on the distance graph (adjacency matrix) with values in the ``[0,inf]`` interval.
	
	.. math::
		c_{ij} = min_{k}( metric ( a_{ik} , b_{kj} ) )

	Args:
		D (matrix): A numpy dense matrix.
		kind (string): type of closure to compute: ``metric`` or ``ultrametric``.
		verbose (bool): Prints statements as it computes.

	Returns:
		C (matrix): transitive closure dense matrix

	Warnings:
		This algorithm uses dense matrix calculations which is still slow for very large graphs.
	
	Examples:

		>>> P = np.array([
			[1.,.9,.1,0.],
			[.9,1.,.8,0.],
			[.1,.8,1.,.6],
			[0.,0.,.6,1.],
			], dtype=float)
		>>> D = prox2dist(P)
		
		>>> transitive_closure(D, kind='metric', verbose=True)
			[ 0. ,.11111111, 0.36111111, 1.02777778],
			[ 0.11111111, 0., 0.25, 0.91666667],
			[ 0.36111111, 0.25, 0., 0.66666667],
			[ 1.02777778, 0.91666667, 0.66666667,  0.]

	Note:
		Metric: :math:`(min,+)`
		
		Ultrametric: :math:`(min,max)` -- also known was maximum flow.
		
		Semantic proximity: TODO
		
		.. math::
			
				[ 1 + \\sum_{i=2}^{n-1} log k(v_i) ]^{-1}


	"""
	_check_for_metric_type(kind)
	
	# Numpy object
	if (type(D).__module__ == np.__name__):
		return _transitive_closure_numpy(D, kind, *args, **kwargs)
	else:
		raise TypeError("Input is not a numpy object")

def _transitive_closure_numpy(A, kind='metric', verbose=False):
	"""
	Calculates Transitive Closure ising numpy dense matrix traversing.
	"""
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
	"""
	TODO
	"""
	# TODO	
	return None
	n,m = A.shape

	for source in xrange(0,n):
		_transitive_closure_dijkstra_matrix(A, source, kind)

def _transitive_closure_dijkstra_matrix(A, source, kind):
	"""
	TODO
	"""
	# TODO
	pass

def _check_for_metric_type(kind):
	"""
	Check for available metric functions.
	"""
	if kind not in __metrics__:
		raise TypeError("Metric not found for this algorithm. Try 'metric' or 'ultrametric',")

