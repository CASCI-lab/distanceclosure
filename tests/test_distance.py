from distanceclosure.distance import pairwise_proximity, _jaccard_coef_bitwise, _jaccard_coef_set, _jaccard_coef_weighted_numpy
import numpy as np
from scipy.sparse import csr_matrix

B = np.array([
	[1,1,1,1],
	[1,1,1,0],
	[1,1,0,0],
	[1,0,0,0],
])

W = np.array([
	[4,3,2,1],
	[3,2,1,0],
	[2,1,0,0],
	[1,0,0,0],
])

def test_jaccard_bitwise():
	""" Test Jaccard: bitwise coef """
	u = np.array([1,1,1,1])
	v = np.array([1,1,1,0])
	d = _jaccard_coef_bitwise(u,v)
	assert (d == 0.75)
	
def test_jaccard_set():
	""" Test Jaccard: set coef """
	u = np.array([4,3,2,1])
	v = np.array([3,2,1,0])
	d = _jaccard_coef_set(u,v)
	assert (d == 0.6)

def test_jaccard_weighted(min_support=1):
	""" Test Jaccard: weighted coef """
	u = np.array([4,3,2,1])
	v = np.array([3,2,1,0])
	d = _jaccard_coef_weighted_numpy(u,v,1)
	print
	assert (d == 0.6)

def test_pairwise_distance_numpy_bitwise():
	""" Test pairwise distance: using the Numpy (dense matrix) implementation for jaccard bitwise coef """
	D = pairwise_proximity(B, metric='jaccard_bitwise')
	true = np.array([
		[ 1.,          0.75,        0.5,         0.25      ],
		[ 0.75,        1.,          0.66666667,  0.33333333],
		[ 0.5,         0.66666667,  1.,          0.5       ],
		[ 0.25,        0.33333333,  0.5,         1.        ],
		], dtype=float)
	assert np.isclose(D, true).all()

def test_pairwise_distance_numpy_set():
	""" Test pairwise distance: using the Numpy (dense matrix) implementation for jaccard set coef """
	D = pairwise_proximity(W, metric='jaccard_set')
	true = np.array([
		[ 1.,          0.6,         0.4,         0.2,       ],
		[ 0.6,         1.,          0.75,        0.5,       ],
		[ 0.4,         0.75,        1.,          0.66666667,],
		[ 0.2,         0.5,         0.66666667,  1.,        ],
 		], dtype=float)
	assert np.isclose(D, true).all()

def test_pairwise_distance_numpy_weighted():
	""" Test pairwise distance: using Numpy (dense matrix) using weighted jaccard """
	D = pairwise_proximity(W, metric='weighted_jaccard', min_support=10)
	true = np.array([
		[ 1.,   0.6,  0.3,  0.1],
		[ 0.6,  1.,   0.,   0. ],
		[ 0.3,  0.,   1.,   0. ],
		[ 0.1,  0.,   0.,   1. ],
		], dtype=float)
	assert np.isclose(D, true).all()

def test_pairwise_distance_sparse_bitwise():
	""" Test pairwise distance: using the Scipy (sparse matrix) implementation for jaccard bitwise coef """
	B_sparse = csr_matrix(B)
	D = pairwise_proximity(B_sparse, metric='jaccard_bitwise')
	#print D.todense()
	true = np.array([
		[ 1.,          0.75,        0.5,         0.25      ],
		[ 0.75,        1.,          0.66666667,  0.33333333],
		[ 0.5,         0.66666667,  1.,          0.5       ],
		[ 0.25,        0.33333333,  0.5,         1.        ],
		], dtype=float)
	assert np.isclose(D.todense(), true).all()

def test_pairwise_distance_sparse_set():
	""" Test pairwise distance: using the Scipy (sparse matrix) implementation for jaccard set coef """
	W_sparse = csr_matrix(W)
	D = pairwise_proximity(W_sparse, metric='jaccard_set')
	true = np.array([
		[ 1.,          0.75,        0.5,         0.25      ],
		[ 0.75,        1.,          0.66666667,  0.33333333],
		[ 0.5,         0.66666667,  1.,          0.5       ],
		[ 0.25,        0.33333333,  0.5,         1.        ],
		], dtype=float)
	assert np.isclose(D.todense(), true).all()

def test_pairwise_distance_sparse_weighted():
	""" Test pairwise distance: using the Scipy (sparse matrix) implementation for jaccard weighted coef """
	W_sparse = csr_matrix(W)
	D = pairwise_proximity(W_sparse, metric='jaccard_weighted')
	print D.todense()
	true = np.array([
		[ 1.,   0.6,  0.3,  0.1],
		[ 0.6,  1.,   0.,   0. ],
		[ 0.3,  0.,   1.,   0. ],
		[ 0.1,  0.,   0.,   1. ],
		], dtype=float)
	assert np.isclose(D.todense(), true).all()

