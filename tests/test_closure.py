from nose.tools import *
from distanceclosure.utils import prox2dist, dict2matrix
from distanceclosure.closure import transitive_closure
from distanceclosure.backbone import backbone
import numpy as np
from scipy.sparse import csr_matrix
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
D_sparse = csr_matrix(D)

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

#
# Dense Matrix
#
@raises(ValueError)
def test_dense_transitive_closure_faults_nonzerodiagonal():
	""" Test Dense for non-zero diagonal """
	Dtmp = D.copy()
	Dtmp[0][0] = 1
	transitive_closure(Dtmp, kind='metric', verbose=True)

def test_dense_transitive_closure_metric():
	""" Test Dense Transitive Closure (Metric) """
	Cm = transitive_closure(D, kind='metric', algorithm='dense', verbose=True)
	assert np.isclose(Cm , Cm_true).all()
	

def test_dense_transitive_closure_ultrametric():
	""" Test Dense Transitive Closure (Ultrametric) """
	Cu = transitive_closure(D, kind='ultrametric', algorithm='dense')
	assert np.isclose(Cu, Cu_true).all()


def test_dense_backbone():
	""" Test Dense Backbone return """
	Cm = transitive_closure(D, kind='metric', algorithm='dense')
	Bm = backbone(D, Cm)
	assert np.isclose(Bm, Bm_true).all()

#
# Dijkstra
#
def test_dijkstra_vs_dense_transitive_closure_ultrametric():
	""" Test Dijkstra vs Dense metric comparison """
	C_Dense_um = transitive_closure(D, kind='metric', algorithm='dense')
	C_Djisktra_um = transitive_closure(D, kind='metric', algorithm='dijkstra')	
	assert (C_Dense_um == dict2matrix(C_Djisktra_um)).all()

def test_dijkstra_vs_dense_transitive_closure_ultrametric():
	""" Test Dijkstra vs Dense ultra metric comparison """
	C_Dense_um = transitive_closure(D, kind='ultrametric', algorithm='dense')
	C_Djisktra_um = transitive_closure(D_sparse, kind='ultrametric', algorithm='dijkstra')	
	assert (C_Dense_um == C_Djisktra_um.A).all()

def test_dijkstra_vs_dense_backbone():
	""" Test Dijkstra vs Dense backbone return """ 
	C_Dense_m = transitive_closure(D, kind='metric', algorithm='dense')
	B_Dense_m = backbone(D, C_Dense_m)

	C_Djisktra_m = transitive_closure(D, kind='metric', algorithm='dijkstra')
	B_Djisktra_m = backbone(D_sparse, C_Djisktra_m)

	# The Sparse matrix version does not put a -1 in the diagonal.
	B_Djisktra_m = B_Djisktra_m.A
	np.fill_diagonal( B_Djisktra_m, -1)
	assert (B_Dense_m == B_Djisktra_m).all()


