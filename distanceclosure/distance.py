# -*- coding: utf-8 -*-
"""
Compute the Jaccard coefficient (distance/similarity) over a matrix

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
	Calculates pairwise proximity coefficient between rows of a matrix

	Parameters
	----------
	M : matrix, numpy or scipy

    metric : string
    	type of jaccard proximity to use.
    	- Binary comparison = 'jaccard','jb','jaccard_bitwise' =
    	- Set comparison ='jaccard_set','js'
    	- Weighted comparison = 'weighted_jaccard','wj'

    Note: Jaccard distance = ( 1 - Coefficient )

	Returns
	--------
	M : matrix
		The matrix of proximities
	
	Usage
	------
	P = pairwise_proximity(M, metric='jaccard')

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
