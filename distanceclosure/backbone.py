# -*- coding: utf-8 -*-
"""
Backbone Extraction
===================

Extracts the Backbone edges from the original graph and the distance closure computation.
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
__all__ = ['backbone']
#
#
#
def backbone(A, C):
	"""
	Return backbone edges based on the the original graph and its transitive closure.
	By definition, the backbone are the edges that did not change value in the transitive closure computation.

	Args:
		A (matrix): Adjacency matrix from original graph.
		C (matrix): Adjacency matrix from transitive closure graph

	Returns:
		B (array): Adjacency matrix where backbone edges are ``1``, semi-metric edges are ``2``, diagonal is ``-1``, ``0`` otherwise.

	Note:
		Inputs accepts both dense and sparse matrices.

	Examples:
		>>> C = transitive_closure_numpy(A, kind='metric')
		>>> B = backbone_numpy(A, C)
		
		You can then access the backbone edges by using:
		
		>>> import numpy as np
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

