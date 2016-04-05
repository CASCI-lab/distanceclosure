from distanceclosure.fuzzylogic import operation
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

G = np.array([
	[0,0,4,3],
	[0,0,2,0],
	[4,2,0,0],
	[3,0,0,0],
])
nodes_G = ['a','b','c','f']
dfG = pd.DataFrame(G, index=nodes_G, columns=nodes_G)
spG = csr_matrix(G)

H = np.array([
	[0,0,2,1],
	[0,0,1,0],
	[2,1,0,0],
	[1,0,0,0],
])
nodes_H = ['a','b','c','d']
dfH = pd.DataFrame(H, index=nodes_H, columns=nodes_H)
spH = csr_matrix(H)

G_inter_H = np.array([
	[0,0,2,0,0],
	[0,0,1,0,0],
	[2,1,0,0,0],
	[0,0,0,0,0],
	[0,0,0,0,0]
])
sp_G_inter_H = csr_matrix(G_inter_H)
G_union_H = np.array([
	[0,0,4,1,3],
	[0,0,2,0,0],
	[4,2,0,0,0],
	[1,0,0,0,0],
	[3,0,0,0,0]	
])
G_diff_H = np.array([
	[0,0,2,-1,3],
	[0,0,1,0,0],
	[2,1,0,0,0],
	[-1,0,0,0,0],
	[3,0,0,0,0]	
])
G_add_H = np.array([
	[0,0,6,1,3],
	[0,0,3,0,0],
	[6,3,0,0,0],
	[1,0,0,0,0],
	[3,0,0,0,0]	
])
nodes_GH = ['a','b','c','d','f']

#
# Numpy Dense (interserction, union, diff & add)
#
def test_dense_G_inter_H():
	""" Test FuzzyLogic: Numpy for Intersection (G \intersection H) """
	N, nodes = operation(G,H,'intersection', nodes_G, nodes_H)
	assert np.isclose(N.values , G_inter_H).all()

def test_dense_G_union_H():
	""" Test FuzzyLogic: Numpy for Union (G \union H) """
	N, nodes = operation(G,H,'union', nodes_G, nodes_H)
	assert np.isclose(N.values , G_union_H).all()

def test_dense_G_diff_H():
	""" Test FuzzyLogic: Numpy for Difference (G \diff H) """
	N, nodes = operation(G,H,'diff', nodes_G, nodes_H)
	assert np.isclose(N.values , G_diff_H).all()	

def test_dense_G_add_H():
	""" Test FuzzyLogic: Numpy for Addition (G \ add H) """
	N, nodes = operation(G,H,'add', nodes_G, nodes_H)
	assert np.isclose(N.values , G_add_H).all()
#
# DataFrame
#
def test_dataframe():
	""" Test FuzzyLogic: DataFrame """
	N, nodes = operation(dfG, dfH,'intersection')
	assert np.isclose(N.values , G_inter_H).all()
#
# Scipy Sparse
#
def test_sparse_G_inter_H():
	""" Test FuzzyLogic: Sparse for Intersection (G \intersection H) """
	N, nodes = operation(spG,spH,'intersection', nodes_G, nodes_H)
	assert np.isclose(N.A , sp_G_inter_H.A).all()

