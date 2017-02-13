# -*- coding: utf-8 -*-
"""
Utils
==========================

Utility functions for the Distance Closure package
"""
#    Copyright (C) 2015 by
#    Luis Rocha <rocha@indiana.edu>
#    Thiago Simas <@.>
#    Rion Brattig Correia <rionbr@gmail.com>
#    All rights reserved.
#    MIT license.
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

__author__ = """\n""".join([
	'Luis Rocha <rocha@indiana.com>',
	'Thiago Simas <@.>',
	'Rion Brattig Correia <rionbr@gmail.com>',
	])

__all__ = [
			'dist2prox',
			'prox2dist',
			'dict2matrix',
			'matrix2dict',
			'dict2sparse'
		]

#
# Proximity and Distance Conversions
#
def prox2dist(P):
	"""
	Transforms a matrix of non-negative ``[0,1]`` proximities P to distance weights in the ``[0,inf]`` interval:
	
	.. math::

		d = \\frac{1}{p} - 1
	
	Args:
		P (matrix): Proximity matrix
	
	Returns:
		D (matrix): Distance matrix

	See Also:
		:attr:`dist2prox`
	"""
	if (type(P).__module__.split('.')[0] == 'numpy'):
		return _prox2dist_numpy(P)
	elif (type(P).__module__.split('.')[1] == 'sparse'):
		return _prox2dist_sparse(P)
	else:
		raise ("Format not accepted: try numpy or scipy.sparse formats")


def _prox2dist_sparse(A):
	A.data = prox2dist_numpy(A.data)
	return A

def _prox2dist_numpy(A):

	f = np.vectorize(_prox2dist)
	return f(A)

def _prox2dist(x):
	if x == 0:
		return np.inf
	else:
		return (1/float(x)) - 1

def dist2prox(D):
	"""
	Transforms a matrix of non-negative integer distances ``D`` to proximity/similarity weights in the ``[0,1]`` interval:
	
	.. math::
	
		p = \\frac{1}{(d+1)}

	It accepts both dense and sparse matrices.

	Args:
		D (matrix): Distance matrix

	Returns:
		P (matrix): Proximity matrix

	See Also:
		:attr:`prox2dist`

	"""
	if (type(D).__module__.split('.')[0] == 'numpy'):
		return _dist2prox_numpy(D)
	elif (type(D).__module__.split('.')[1] == 'sparse'):
		return _dist2prox_numpy(D)
	else:
		raise ValueError("Format not accepted: try numpy or scipy.sparse formats")

def _dist2prox_sparse(A):
	A.data = _dist2prox_numpy(A.data)
	return A

def _dist2prox_numpy(A):

	f = np.vectorize(_dist2prox)
	return f(A)

def _dist2prox(x):
	if x == np.inf:
		return 0
	else:
		return (x + 1) ** -1
#
# Data format Conversiosn
#
def dict2matrix(d):
	"""
	Tranforms a 2D dictionary into a numpy. Usefull when converting Dijkstra results.

	Args:
		d (dict): 2D dictionary

	Returns:
		m (matrix): numpy matrix

	Warning:
		If your nodes have names instead of number assigned to them, make sure to keep a mapping.

	Usage:
		>>> d = {0: {0: 0, 1: 1, 2:3}, 1: {0: 1, 1: 0, 2:2}, 2: {0: 3, 1:2, 2:0}}
		>>> dict2matrix(d)
		[[ 0 1 3]
		 [ 1 0 2]
		 [ 3 2 0]]

	See Also:
		:attr:`matrix2dict`

	Note:
		Uses pandas to accomplish this in a one liner.
	"""
	return pd.DataFrame.from_dict(d).values

def matrix2dict(m):
	"""
	Tranforms a Numpy matrix into a 2D dictionary. Usefull when comparing dense metric and Dijkstra results.

	Args:
		m (matrix): numpy matrix

	Returns:
		d (dict): 2D dictionary

	Usage:
		>>> m = [[0, 1, 3], [1, 0, 2], [3, 2, 0]]
		>>> matrix2dict(m)
		{0: {0: 0, 1: 1, 2:3}, 1: {0: 1, 1: 0, 2:2}, 2: {0: 3, 1:2, 2:0}}

	See Also:
		:attr:`dict2matrix`

	Note:
		Uses pandas to accomplish this in a one liner.
	"""
	df = pd.DataFrame(m)
	return pd.DataFrame(m).to_dict()

def dict2sparse(d):
	"""
	Tranforms a 2D dictionary into a Scipy sparse matrix.

	Args:
		d (dict): 2D dictionary

	Returns:
		m (csr matrix): CRS Sparse Matrix

	Usage:
		>>> d = {0: {0: 0, 1: 1, 2:3}, 1: {0: 1, 1: 0, 2:2}, 2: {0: 3, 1:2, 2:0}}
		>>> dict2sparse(d)
		(0, 1)	1
		(0, 2)	3
		(1, 0)	1
		(1, 2)	2
		(2, 0)	3
		(2, 1)	2

	See Also:
		:attr:`dict2matrix`, :attr:`matrix2dict`

	Note:
		Uses pandas to convert dict into dataframe and then feeds it to the `csr_matrix`.
	"""
	return csr_matrix(pd.DataFrame.from_dict(d).values)

