from nose.tools import *
from distanceclosure.utils import prox2dist
from distanceclosure.closure import transitive_closure
from distanceclosure.backbone import backbone
import numpy as np
#
# Test
#
# Proximity Network
P = np.array([
		[1.,.9,.1,0.],
		[.9,1.,.8,0.],
		[.1,.8,1.,.6],
		[0.,0.,.6,1.],
		], dtype=float)
D = prox2dist(P)

Cm_true = np.array([
		[ 0. ,.11111111, 0.36111111, 1.02777778],
		[ 0.11111111, 0., 0.25, 0.91666667],
		[ 0.36111111, 0.25, 0., 0.66666667],
		[ 1.02777778, 0.91666667, 0.66666667,  0.]
		], dtype=float)

Cu_true = np.array([
		[ 0. ,.11111111, 0.25, 0.66666667],
		[ 0.11111111, 0., 0.25, 0.66666667],
		[ 0.25, 0.25, 0., 0.66666667],
		[ 0.66666667, 0.66666667, 0.66666667,  0.]
	], dtype=float)

Bm_true = np.array([
		[-1,1,2,2],
		[1,-1,1,2],
		[2,1,-1,1],
		[2,2,1,-1],
		], dtype=int)

@raises(ValueError)
def test_transitive_closure_faults_nonzeroentries():
	""" Test for distance matrix with zero entries """
	Dtmp = D.copy()
	Dtmp[0][1] = 0
	transitive_closure(Dtmp, kind='metric', verbose=True)

@raises(ValueError)
def test_transitive_closure_faults_nonzerodiagonal():
	""" Test for non-zero diagonal """
	Dtmp = D.copy()
	Dtmp[0][0] = 1
	transitive_closure(Dtmp, kind='metric', verbose=True)

def test_transitive_closure_metric():
	""" Test Transitive Closure (Metric) """
	Cm = transitive_closure(D, kind='metric', verbose=True)
	assert np.isclose(Cm , Cm_true).all()
	

def test_transitive_closure_ultrametric():
	""" Test Transitive Closure (Ultrametric) """
	Cu = transitive_closure(D, kind='ultrametric')
	assert np.isclose(Cu, Cu_true).all()


def test_backbone():
	""" Test the Backbone return """
	Cm = transitive_closure(D, kind='metric')
	Bm = backbone(D, Cm)
	assert np.isclose(Bm, Bm_true).all()




