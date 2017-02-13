# -*- coding: utf-8 -*-
"""
Pairwise distance
====================

Computes the Jaccard proximity over a matrix

"""
#    Copyright (C) 2015 by
#    Rion Brattig Correia <rionbr@gmail.com>
#    All rights reserved.
#    MIT license.
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial.distance import cdist, squareform, jaccard
from itertools import combinations
import warnings

__author__ = """\n""".join(['Rion Brattig Correia <rionbr@gmail.com>'])
__all__ = ['pairwise_proximity']
__metrics__ = [
				'jaccard','scipy', # Numeric Jaccard (scipy.spatial.distance)
				'jaccard_binary','jb', # Binary Jaccard Coefficient
				'jaccard_set','js', # Set Comparison Jaccard Coefficient
				'jaccard_weighted','weighted_jaccard','wj', # Weighted Jaccard
				]

def pairwise_proximity(M, metric='jaccard', *args, **kwargs):
	"""
	Calculates pairwise proximity coefficient between rows of a matrix.
	Three types of Jaccard proximity is available depending on your data.
	
	Args:
		M (matrix) : adjacency matrix
		metric (str) : Jaccard proximity metric
			Allowed values:
				- Binary item-wise comparison: ``jaccard_binary``, ``jb``
				- Numeric item-wise comparison: ``jaccard``, (scipy.spatial.dist.jaccard)
				- Set comparison: ``jaccard_set``, ``js``
				- Weighted item-wise comparison: ``weighted_jaccard``, ``wj``
				- Note: Also accepts a custom function being passed.
		min_support (Optional[int]) : the minimum support passed to the metric function.
		verbose (bool) : print every line as it computes.

	Returns:
		M (matrix) : The matrix of proximities
	
	Examples:

		There are four ways to compute the proximity, here are some examples:

		>>> # Numeric Matrix (not necessarily a network)
		>>> N = np.array([
			[2,3,4,2],
			[2,3,4,2],
			[2,3,3,2],
			[2,1,3,4]])

		>>> # Binary Adjacency Matrix
		>>> B = np.array([
			[1,1,1,1],
			[1,1,1,0],
			[1,1,0,0],
			[1,0,0,0]])
		
		>>> # Weighted Adjacency Matrix
		>>> W = np.array([
			[4,3,2,1],
			[3,2,1,0],
			[2,1,0,0],
			[1,0,0,0]])
		

		Numeric Jaccard: the default and most commonly used version. Implemented from `scipy.spatial.distance`.
		
		>>> pairwise_proximity(N, metric='jaccard')
			[[ 1. , 1.  , 0.75, 0.25],
			[ 1.  , 1.  , 0.75, 0.25],
			[ 0.75, 0.75, 1.  , 0.5 ],
			[ 0.25, 0.25, 0.5 , 1.  ]]

		Binary Jaccard: the default and most commonly used version.

		>>> pairwise_proximity(B, metric='jaccard_binary')
			[[ 1. , 0.75, 0.5 , 0.25],
			[ 0.75, 1.  , 0.66, 0.33],
			[ 0.5 , 0.66, 1.  , 0.5 ],
			[ 0.25, 0.33, 0.5 , 1.  ]]

		Set Jaccard: it treats the values in each vector as a set of objects, therefore their order is not taken into account.
		Note that zeroes are treated as a set item.
		
		>>> pairwise_proximity(B, metric='jaccard_set')
			[[ 1., 0.6 , 0.4 , 0.2 ],
			[ 0.6, 1.  , 0.75, 0.5 ],
			[ 0.4, 0.75, 1.  , 0.67],
			[ 0.2, 0.5 , 0.67, 1.  ]]

		Weighted Jaccard: the version for weighted graphs.

		>>> pairwise_proximity(W, metric='jaccard_weighted')
			[ 1.,   0.6,  0.3,  0.1],
			[ 0.6,  1.,   0.,   0. ],
			[ 0.3,  0.,   1.,   0. ],
			[ 0.1,  0.,   0.,   1. ],
	"""

	# Numpy object
	if (type(M).__module__ == np.__name__):
		return _pairwise_proximity_numpy(M, metric, *args, **kwargs)
	elif (sp.issparse(M)):
		return _pairwise_proximity_sparse(M, metric, *args, **kwargs)
	else:
		raise TypeError("Input is not a valid object, try a numpy array")


def _pairwise_proximity_numpy(M, metric='jaccard', *args, **kwargs):	
	""" Pairwise proximity computation over dense matrix (numpy) """
	
	# If is not a Numpy array
	if type(M) != 'numpy.ndarray':
		M = np.array(M)

	# If matrix has negative entries
	if M.min() < 0:
		raise TypeError("Matrix cannot have negative numbers")

	# Get coef (metric) function from string
	if isinstance(metric, str):
		coef = _get_dense_metric_function(metric)
	# or a function was passed
	else:
		coef = metric

	# Verbose Attr
	verbose = kwargs.pop('verbose', False)

	# Calculate proximity
	m, n = M.shape
	pm = np.zeros((m * (m - 1)) // 2, dtype=np.double)
	
	k = 0
	for i in xrange(0, m - 1):
		if verbose:
			print 'calc row:',i,'of',m-1
		for j in xrange(i + 1, m):
			pm[k] = coef(M[i,:], M[j,:], *args, **kwargs)
			k += 1
	
	pm = squareform(pm) #Make into a matrix format
	np.fill_diagonal(pm, 1) #Fill diagonal
	return pm

def _pairwise_proximity_sparse(X, metric='jaccard', *args, **kwargs):
	""" Pairwise proximity computation over sparse matrix (scipy.sparse) """

	# Get coef (metric) function from string
	if isinstance(metric, str):
		how, coef = _get_sparse_metric_function(metric)
	# or a function was passed
	else:
		how, coef = metric
	
	if how == 'indices':
		generator = (coef(row1.indices, row2.indices) for row1, row2 in combinations(X, r=2))
	elif how == 'values':
		generator = (coef(row1.data, row2.data) for row1, row2 in combinations(X, r=2))
	elif how == 'both':
		generator = (coef(row1.toarray(), row2.toarray()) for row1, row2 in combinations(X, r=2))

	S_flattened = np.fromiter(generator, dtype=np.float64)
	S = squareform(S_flattened)
	S = lil_matrix(S)
	S.setdiag(1, k=0)
	return S.tocsr()

def _jaccard_coef_scipy(u, v, min_support=1):
	if np.sum(u) + np.sum(v) >= min_support:
		return 1-jaccard(u,v)
	else:
		return 0.

def _jaccard_coef_binary(u, v, min_support=1):
	u = u.astype(bool)
	v = v.astype(bool)
	if np.sum(u + v) >= min_support:
		return (np.double(np.bitwise_and(u,v).sum()) / np.double( np.bitwise_or(u,v).sum()))
	else:
		return 0.

def _jaccard_coef_set(u, v, min_support=1):
	u = set(u)
	v = set(v)
	inter_len = len(u.intersection(v))
	union_len = len(u) + len(v) - inter_len
	if union_len >= min_support:
		return np.double(inter_len) / np.double(union_len)
	else:
		return 0.

def _jaccard_coef_weighted_numpy(u,v,min_support=10):
	VMax = np.maximum(u,v) #Find maximum Values
	if np.sum(VMax) >= min_support: #Only compute when MAX()
		VMin = np.minimum(u,v) #Find minimum Values
		sumMin = np.sum(VMin) #Sum values
		sumMax = np.sum(VMax)
		coef = np.true_divide(sumMin,sumMax) # (Sum Min) / (Sum Max)
		return coef
	else:
		return 0.

def _check_for_metric_type(metric):
	if metric not in __metrics__:
		raise TypeError("Metric kind should be one of: '" + "' '".join(__metrics__)+ "'")

def _get_dense_metric_function(metric):
	
	_check_for_metric_type(metric)

	if metric in ['jaccard','scipy','j']:
		# in practice it computes the dissimilarity, so 1-dissimilarity
		return _jaccard_coef_scipy

	elif metric in ['jaccard_binary','jb']:
		return _jaccard_coef_binary

	elif metric in ['jaccard_set','js']:
		warnings.warn('Zeros will be considered as a item set.')
		return _jaccard_coef_set

	elif metric in ['jaccard_weighted','weighted_jaccard','wj']:
		return _jaccard_coef_weighted_numpy

def _get_sparse_metric_function(metric):

	_check_for_metric_type(metric)

	if metric in ['jaccard','scipy']:
		# in practice it computes the dissimilarity, so 1-dissimilarity
		return ('values', _jaccard_coef_scipy)

	if metric in ['jaccard_binary','jb']:
		return ('indices', _jaccard_coef_set)

	elif metric in ['jaccard_set','js']:
		return ('values', _jaccard_coef_set)

	elif metric in ['jaccard_weighted','weighted_jaccard','wj']:
		return ('both', _jaccard_coef_weighted_numpy)
