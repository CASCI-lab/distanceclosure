from nose.tools import raises
from distanceclosure.utils import prox2dist, dict2matrix
from distanceclosure.closure import distance_closure
import numpy as np
from scipy.sparse import csr_matrix
#
# Test
#
# Proximity Network
P = np.array([
             [1., .9, .1, 0.],
             [.9, 1., .8, 0.],
             [.1, .8, 1., .6],
             [0., 0., .6, 1.]], dtype=float)
D = np.vectorize(prox2dist)(P)
D_sparse = csr_matrix(D)

Cm_true = np.array([
                   [0., .11111111, 0.36111111, 1.02777778],
                   [0.11111111, 0., 0.25, 0.91666667],
                   [0.36111111, 0.25, 0., 0.66666667],
                   [1.02777778, 0.91666667, 0.66666667, 0.]], dtype=float)

Cu_true = np.array([
                   [0., .11111111, 0.25, 0.66666667],
                   [0.11111111, 0., 0.25, 0.66666667],
                   [0.25, 0.25, 0., 0.66666667],
                   [0.66666667, 0.66666667, 0.66666667, 0.]], dtype=float)

Bm_true = np.array([
                   [-1, 1, 2, 2],
                   [1, -1, 1, 2],
                   [2, 1, -1, 1],
                   [2, 2, 1, -1]], dtype=int)


#
# Dense Matrix
#
@raises(ValueError)
def test_dense_distance_closure_faults_nonzerodiagonal():
    """ Test Closure: Dense for non-zero diagonal """
    Dtmp = D.copy()
    Dtmp[0][0] = 1
    distance_closure(Dtmp, kind='metric', algorithm='dense', verbose=True)


def test_dense_distance_closure_metric():
    """ Test Closure: Dense Transitive Closure (Metric) """
    Cm = distance_closure(D, kind='metric', algorithm='dense', verbose=True)
    assert np.isclose(Cm, Cm_true).all()


def test_dense_distance_closure_ultrametric():
    """ Test Closure: Dense Transitive Closure (Ultrametric) """
    Cu = distance_closure(D, kind='ultrametric', algorithm='dense')
    assert np.isclose(Cu, Cu_true).all()


#
# Dijkstra
#
def test_dijkstra_vs_dense_distance_closure_metric():
    """ Test Closure: Dijkstra vs Dense metric comparison """
    C_Dense_um = distance_closure(D, kind='metric', algorithm='dense')
    C_Djisktra_um = distance_closure(D, kind='metric', algorithm='dijkstra')

    assert (C_Dense_um == C_Djisktra_um.A).all()


def test_dijkstra_vs_dense_distance_closure_ultrametric():
    """ Test Closure: Dijkstra vs Dense ultrametric comparison """
    C_Dense_um = distance_closure(D, kind='ultrametric', algorithm='dense')
    C_Djisktra_um = distance_closure(D_sparse, kind='ultrametric', algorithm='dijkstra')
    assert (C_Dense_um == C_Djisktra_um.A).all()
