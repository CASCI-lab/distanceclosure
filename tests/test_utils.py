from distanceclosure.utils import prox2dist, dist2prox
from distanceclosure.utils import dict2matrix, matrix2dict, dict2sparse
import numpy as np
from scipy.sparse import csr_matrix

P = P_true = np.array([
		[1.,.9,.1,0.],
		[.9,1.,.8,0.],
		[.1,.8,1.,.6],
		[0.,0.,.6,1.],
		], dtype=float)

D = D_true = np.array([
		[0.,.11111111,9.,np.inf],
		[.11111111,0.,0.25,np.inf],
		[9.,0.25,0.,0.66666667],
		[np.inf,np.inf,0.66666667,0.],
		], dtype=float)

#
# Test Distance Proximity Conversion
#
def test_prox2dist():
	""" Test Utils: Prox2Dist """
	assert np.isclose(dist2prox(D), P_true).all()

def test_dist2prox():
	""" Test Utils: Dist2Prox """
	assert np.isclose(prox2dist(P), D_true).all()

def test_dist2prox_prox2dist():
	""" Test Utils: Prox2Dist & Dist2Prox """
	assert np.isclose(dist2prox(prox2dist(P)) , P).all()
#
# Test Data Conversion
#
def test_matrix2dict():
	""" Test Utils: matrix 2 dict """
	m = [[0, 1, 3], [1, 0, 2], [3, 2, 0]]
	d = matrix2dict(m)
	assert (d == {0: {0: 0, 1: 1, 2:3}, 1: {0: 1, 1: 0, 2:2}, 2: {0: 3, 1:2, 2:0}} )

def test_dict2matrix():
	""" test Utils: dict 2 matrix """
	d = {0: {0: 0, 1: 1, 2:3}, 1: {0: 1, 1: 0, 2:2}, 2: {0: 3, 1:2, 2:0}}
	m = dict2matrix(d)
	assert (m == np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]]) ).all()

def test_dict2sparse():
	""" Test Utils: dict 2 sparse """
	d = {0: {0: 0, 1: 1, 2:3}, 1: {0: 1, 1: 0, 2:2}, 2: {0: 3, 1:2, 2:0}}
	s = dict2sparse(d)
	t = csr_matrix(np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]])) 
	assert np.allclose(s.A, t.A)
