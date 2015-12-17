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

__author__ = """\n""".join([
	'Luis Rocha <rocha@indiana.com>',
	'Thiago Simas <@.>',
	'Rion Brattig Correia <rionbr@gmail.com>',
	])

__all__ = [
			'dist2prox',
			'prox2dist',
		]

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
	if (type(P).__module__ == 'numpy'):
		return _prox2dist_numpy(P)
	elif (type(P).__module__.split('.')[1] == 'sparse'):
		return _prox2dist_sparse(P)
	else:
		raise ("Format not accepted: try numpy or scipy.sparse formats")


def _prox2dist_sparse(A):
	A.data = prox2dist_numpy(A.data)
	return A

def _prox2dist_numpy(A):
	
	def _prox2dist(x):
		if x == 0:
			return np.inf
		else:
			return (1/float(x)) - 1

	f = np.vectorize(_prox2dist)
	return f(A)

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
	if (type(D).__module__ == 'numpy'):
		return _dist2prox_numpy(D)
	elif (type(D).__module__.split('.')[1] == 'sparse'):
		return _dist2prox_numpy(D)
	else:
		raise ("Format not accepted: try numpy or scipy.sparse formats")

def _dist2prox_sparse(A):
	A.data = dist2prox_numpy(A.data)
	return A

def _dist2prox_numpy(A):

	def _dist2prox(x):
		if x == np.inf:
			return 0
		else:
			return (x + 1) ** -1

	f = np.vectorize(_dist2prox)
	return f(A)


