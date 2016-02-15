# -*- coding: utf-8 -*-
"""
Backbone extraction
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
import scipy.sparse as ssp
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
		A (dense or sparse matrix) : Adjacency matrix from original graph.
		C (dense or sparse matrix) : Adjacency matrix from transitive closure graph

	Returns:
		B (dense or sparse matrix) : Adjacency matrix where backbone `metric` edges are ``1`` and `semi-metric` edges are ``2``. On the dense matrix, the diagonal is ``-1`` and ``0`` otherwise.

	Examples:

		>>> # Dense Matrix
		>>> C = transitive_closure(A, kind='metric', algorithm='dense')
		>>> B = backbone(A, C)
		
		You can then access the backbone metric edges by using:
		
		>>> import numpy as np
		>>> rows, cols = np.where(B==1)

		>>> # Sparse Matrix
		>>> C = transitive_closure(A, kind='metric', algorithm='dijkstra')
		>>> B = backbone(A, C)
	"""	
	# Check for data type
	if (type(A).__module__ == np.__name__) and (type(C).__module__ == np.__name__):
		return _backbone_numpy(A, C)
	elif (ssp.issparse(A)) and (ssp.issparse(C)):
		return _backbone_sparse(A, C)
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
	# Metric values = 1
	rows, cols = np.where( (C > 0) & (C != np.inf) & np.isclose(A , C) )
	B[rows,cols] = 1
	return B

def _backbone_sparse(A, C):
	n,m = A.shape
	A = A.tocsr()
	C = C.tocoo()

	B_row = []; B_col = []; B_data = []

	for i, j, v in zip(C.row, C.col, C.data):
		try:
			a_v = A[i,j]
		except:
			B_row.append(i)
			B_col.append(j)
			B_data.append(1)
		else:	
			# Metric values = 1
			if ( np.isclose( [A[i,j]] , [v] ) & (v != np.inf) ):
				B_row.append(i)
				B_col.append(j)
				B_data.append(1)
			# Semi-metric values = 2
			elif (v != np.inf):
				B_row.append(i)
				B_col.append(j)
				B_data.append(2)
	B = ssp.coo_matrix((B_data, (B_row, B_col)), shape=(n, m))
	return B.tocsr()
	

