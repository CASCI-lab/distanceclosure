from distanceclosure.utils import prox2dist, dist2prox
import numpy as np

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

def test_prox2dist():
	print '--- Test Prox2Dist ---'
	assert np.isclose(dist2prox(D), P_true).all()

def test_dist2prox():
	print '--- Test Dist2Prox ---'
	assert np.isclose(prox2dist(P), D_true).all()

def test_dist2prox_prox2dist():
	print '--- Test Prox2Dist & Dist2Prox ---'
	assert np.isclose(dist2prox(prox2dist(P)) , P).all()

