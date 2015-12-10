# -*- coding: utf-8 -*-
"""
Utility functions for the Distance Closure package
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
			'dist2prox',
			'prox2dist',
		]

__metrics__ = ['metric','ultrametric','semantic']


def prox2dist(O):
	"""
	Transforms a matrix of non-negative [0,1] proximities x to
	distance weights in the [0,inf] interval:
		      1
		s = ----- - 1
		      x
	Parameters
	----------
	x : array_like
		an array of non-negative distances.
	"""
	if (type(O).__module__ == 'numpy'):
		return prox2dist_numpy(O)
	elif (type(O).__module__.split('.')[1] == 'sparse'):
		return prox2dist_sparse(O)
	else:
		raise ("Format not accepted: try numpy or scipy.sparse formats")


def prox2dist_sparse(A):
	A.data = prox2dist_numpy(A.data)
	return A

def prox2dist_numpy(A):
	
	def _prox2dist(x):
		if x == 0:
			return np.inf
		else:
			return (1/float(x)) - 1

	f = np.vectorize(_prox2dist)
	return f(A)

def dist2prox(O):
	"""
	Transforms a matrix of non-negative integer distances x to
	proximity/similarity weights in the [0,1] interval:
		      1
		s = -----
		    x + 1
	Parameters
	----------
	x : array_like
		an array of non-negative distances.
	"""
	if (type(O).__module__ == 'numpy'):
		return dist2prox_numpy(O)
	elif (type(O).__module__.split('.')[1] == 'sparse'):
		return dist2prox_numpy(O)
	else:
		raise ("Format not accepted: try numpy or scipy.sparse formats")

def dist2prox_sparse(A):
	A.data = dist2prox_numpy(A.data)
	return A

def dist2prox_numpy(A):

	def _dist2prox(x):
		if x == np.inf:
			return 0
		else:
			return (x + 1) ** -1

	f = np.vectorize(_dist2prox)
	return f(A)


