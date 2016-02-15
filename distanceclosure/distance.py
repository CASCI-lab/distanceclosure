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
from scipy.spatial.distance import cdist, squareform
from itertools import combinations
import warnings

__author__ = """\n""".join(['Rion Brattig Correia <rionbr@gmail.com>'])
__all__ = ['pairwise_proximity']
__metrics__ = [
				'jaccard','jb','jaccard_bitwise', # Bitwise Jaccard Coefficient
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
				- Binary item-wise comparison: ``jaccard``, ``jb``, ``jaccard_bitwise``
				- Set comparison: ``jaccard_set``, ``js``
				- Weighted item-wise comparison: ``weighted_jaccard``, ``wj``
		min_support (Optional[int]) : the minimum support passed to 'weighted_jaccard'

	Returns:
		M (matrix) : The matrix of proximities
	
	Examples:

		There are three ways to compute the proximity, here are some examples:

		>>> # Binary Matrix
		>>> B = np.array([
			[1,1,1,1],
			[1,1,1,0],
			[1,1,0,0],
			[1,0,0,0],
		])
		>>> # Weighted Matrix
		>>> W = np.array([
			[4,3,2,1],
			[3,2,1,0],
			[2,1,0,0],
			[1,0,0,0],
		])
		
		Binary Jaccard: the default and most commonly used version.

		>>> pairwise_proximity(B, metric='jaccard')
			[[ 1. , 0.75, 0.5 , 0.25],
			[ 0.75, 1.  , 0.66, 0.33],
			[ 0.5 , 0.66, 1.  , 0.5 ],
			[ 0.25, 0.33, 0.5 , 1.  ]]

		Set Jaccard: it treats the values in each vector as a set of objects, therefore their order is not taken into account.
		
		>>> pairwise_proximity(B, metric='set')
			[[ 1., 0.6 , 0.4 , 0.2 ],
			[ 0.6, 1.  , 0.75, 0.5 ],
			[ 0.4, 0.75, 1.  , 0.67],
			[ 0.2, 0.5 , 0.67, 1.  ]]

		Weighted Jaccard: the version for weighted graphs.

		>>> pairwise_proximity(W, metric='weighted')
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
	coef = _get_dense_metric_function(metric)

	# If is not a Numpy array
	if type(M) != 'numpy.ndarray':
		M = np.array(M)

	# If matrix has negative entries
	if M.min() < 0:
		raise TypeError("Matrix cannot have negative numbers")

	# Calculate proximity
	m, n = M.shape
	dm = np.zeros((m * (m - 1)) // 2, dtype=np.double)
	
	k = 0
	for i in xrange(0, m - 1):
		print 'calc row:',i,'of',m-1
		for j in xrange(i + 1, m):
			dm[k] = coef(M[i,:], M[j,:], *args, **kwargs)
			k += 1
	
	dm = squareform(dm) #Make into a matrix format
	np.fill_diagonal(dm, 1) #Fill diagonal
	return dm

def _pairwise_proximity_sparse(X, metric='jaccard'):
	""" Pairwise proximity computation over sparse matrix (scipy.sparse) """
	how, coef = _get_sparse_metric_function(metric)
	
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

def _jaccard_coef_bitwise(u, v):
	u = u.astype(bool)
	v = v.astype(bool)
	return (np.double(np.bitwise_and(u,v).sum()) / np.double( np.bitwise_or(u,v).sum()))

def _jaccard_coef_set(u, v):
	u = set(u)
	v = set(v)
	inter_len = len(u.intersection(v))
	union_len = len(u) + len(v) - inter_len
	return np.double(inter_len) / np.double(union_len)

def _jaccard_coef_weighted_numpy(u,v,min_support=10):
	VMax = np.maximum(u,v) #Find maximum Values
	if np.sum(VMax) >= min_support: #Only compute when MAX()
		VMin = np.minimum(u,v) #Find minimum Values
		sumMin = np.sum(VMin) #Sum values
		sumMax = np.sum(VMax)
		coef = np.true_divide(sumMin,sumMax) # (Sum Min) / (Sum Max)
		return coef
	else:
		return 0

def _check_for_metric_type(metric):
	if metric not in __metrics__:
		raise TypeError("Metric kind should be one of: '" + "' '".join(__metrics__)+ "'")

def _get_dense_metric_function(metric):
	
	_check_for_metric_type(metric)

	if metric in ['jaccard','jb','jaccard_bitwise']:
		return _jaccard_coef_bitwise

	elif metric in ['jaccard_set','js']:
		warnings.warn('Zeros will be considered as a item set.')
		return _jaccard_coef_set

	elif metric in ['weighted_jaccard','wj']:
		return _jaccard_coef_weighted_numpy

def _get_sparse_metric_function(metric):

	_check_for_metric_type(metric)

	if metric in ['jaccard','jb','jaccard_bitwise']:
		return ('indices', _jaccard_coef_set)

	elif metric in ['jaccard_set','js']:
		return ('values', _jaccard_coef_set)

	elif metric in ['jaccard_weighted','weighted_jaccard','wj']:
		return ('both', _jaccard_coef_weighted_numpy)
