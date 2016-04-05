# -*- coding: utf-8 -*-
"""
Fuzzy Logic for graphs
====================

Operations of fuzzy logic between graphs.
Notation: given two graphs :math:`G` and :math:`H`, calculates a fuzzy function between them.

.. math::
	N = f(G, H)

Where :math:`N` is the new graph formed by :math:`f(G,H)`.

Note:
	The currently supports functions are: ``union``, ``intersection``, ``addition`` and ``subtraction``.

"""
#    Copyright (C) 2015 by
#    Rion Brattig Correia <rionbr@gmail.com>
#    All rights reserved.
#    MIT license.
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix
import warnings

__author__ = """\n""".join(['Rion Brattig Correia <rionbr@gmail.com>'])
__all__ = ['operation']
__operations__ = [
	'union','U','u',
	'intersection','i',
	'difference','diff','d',
	'addition','add','a'
	]

def operation(G, H, operation, nodes_G=None, nodes_H=None, *args, **kwargs):
	"""
	Applies a fuzzy logic operation to two graphs.
	
	Args:
		G (matrix) : adjacency matrix
		H (matrix) : adjacency matrix
		operation (str) : Fuzzy logic operation to be computed.
			
			Allowed values:
				- Union (:math:`max(G, H)`): ``union``, ``U``, ``u``
				- Intersection (:math:`min(G, H)`): ``intersection``, ``i``
				- Difference (:math:`G - H`): ``difference``, ``diff``, ``d``
				- Addition (:math:`G + H`): ``addition``, ``add``, ``a``
		nodes_G (list) : node's unique identifiers for matrix G, required if G is a Numpy dense or Scipy.sparse matrix.
		nodes_H (list) : node's unique identifiers for matrix H, required if H is a Numpy dense or Scipy.sparse matrix.

	Returns:
		M (matrix) : The computed adjacency matrix. If input is dense, as a Pandas DataFrame object otherwise a Scipy.sparse object.
		nodes (list) : A list of the new combined nodes.
	
	Examples:
		
		Calculate the fuzzy intersection between two graphs, ``G`` and ``H``.
		
		>>> G = np.array([
			[0,0,4,3],
			[0,0,2,0],
			[4,2,0,0],
			[3,0,0,0],
		])
		>>> nodes_G = ['a','b','c','f']
		>>> H = np.array([
			[0,0,2,1],
			[0,0,1,0],
			[2,1,0,0],
			[1,0,0,0],
		])
		>>> nodes_H = ['a','b','c','d']
		>>> N, nodes = operation(G, H, 'intersection', nodes_G, nodes_H)
		>>> N
			[0,0,2,0,0],
			[0,0,1,0,0],
			[2,1,0,0,0],
			[0,0,0,0,0],
			[0,0,0,0,0]
		>>> nodes = ['a','b','c','d','f']

	Note:
		Accepts Pandas DataFrame, Numpy dense and Scipy.sparse objects as inputs.
		
		For Numpy dense and Scipy.sparse, node_ids must be specified (``nodes_G`` & ``nodes_H``).
		For DataFrames, node_ids will be extracted from index.
	"""

	assert operation in __operations__, 'Operation parameter must be one of the following "%s". Received "%s"' % ( ', '.join(__operations__), operation )

	# Numpy object
	if ((type(G).__module__ == np.__name__) and (type(H).__module__ == np.__name__)):

		return _operation(G, H, operation, 'dense', nodes_G, nodes_H, *args, **kwargs)
	
	# Scipy.sparse object
	elif ( (sp.issparse(G)) and (sp.issparse(H)) ):

		return _operation(G, H, operation, 'sparse', nodes_G, nodes_H, *args, **kwargs)

	elif ( (isinstance(G, pd.DataFrame)) and (isinstance(H, pd.DataFrame)) ):

		return _operation_dataframe(G, H, operation, *args, **kwargs)

	else:
		raise TypeError("Input is not a valid object, try a Numpy, Scipy.sparse array or a Pandas DataFrame. Be consistent with types.")

def _operation_dataframe(G, H, operation, *args, **kwargs):
	""" Calculates the requested operation on the Pandas DataFrame networks."""
	nodes_G = G.index
	nodes_H = H.index
	G = G.values
	H = H.values
	return _operation(G, H, operation, 'dense', nodes_G, nodes_H, args, kwargs)

def _operation(G, H, operation, computation_type=None, nodes_G=None, nodes_H=None, *args, **kwargs):	
	""" Calculates the requested operation on the Numpy dense networks.

	Note:
		Works with both undirected and directed networks
	"""
	# assertions
	assert (nodes_G is not None) , 'Nodes G is empty. For dense matrices you must specify node unique identifiers.'
	assert (nodes_H is not None) , 'Nodes H is empty. For dense matrices you must specify node unique identifiers.'
	assert (G.shape[0] == G.shape[1]) , 'Matrix G is not square.'
	assert (G.shape[0] == G.shape[1]) , 'Matrix H is not square.'
	assert (len(nodes_G) == G.shape[0]) , 'Both G and Nodes G must have the same dimensions.'
	assert (len(nodes_H) == H.shape[0]) , 'Both H and Nodes H must have the same dimensions.'

	_op = _get_operation_from_string(operation)

	# Make new list that have all elements
	nodes = sorted(list(nodes_G) + list(set(nodes_H) - set(nodes_G)))
	n = len(nodes)
	
	if computation_type == 'dense':
		# Transform everything to dicts
		dG = {(i_l,j_l):cell for i,(i_l,row) in enumerate(zip(nodes_G, G)) for j,(j_l,cell) in enumerate(zip(nodes_G, row))}
		dH = {(i_l,j_l):cell for i,(i_l,row) in enumerate(zip(nodes_H, H)) for j,(j_l,cell) in enumerate(zip(nodes_H, row))}
	elif computation_type == 'sparse':
		G = G.tocoo()
		H = H.tocoo()
		dG = {(nodes_G[i],nodes_G[j]):v for i,j,v in zip(G.row, G.col, G.data)}
		dH = {(nodes_H[i],nodes_H[j]):v for i,j,v in zip(H.row, H.col, H.data)}
	else:
		# Sanity check
		raise TypeError('Problem with type of computation requested.')

	
	# New Dict that contains all possible edges from both networks
	dN = dG.copy()
	dN.update(dH)
	dN = dict.fromkeys(dN, 0)

	# Iterate over all possible edges
	for n1, n2 in dN.keys():
		# Exists in G
		if ( (n1 in nodes_G) and (n2 in nodes_G) ):
			eG = dG[(n1,n2)]
		else:
			eG = 0

		# Exists in H
		if ( (n1 in nodes_H) and (n2 in nodes_H) ):
			eH = dH[(n1,n2)]
		else:
			eH = 0
		
		# apply fuzzy operation
		dN[(n1,n2)] = _op( eG , eH )


	# Create a MultiIndex, creates a DataFrame with the MultiIndex, then unstacks it.
	if computation_type == 'dense':
		M = pd.DataFrame(dN.values(), index=pd.MultiIndex.from_tuples(dN.keys()))
		M = M.unstack(level=-1, fill_value=0)
		M.columns = M.columns.droplevel()
	# Creates as new Scipy.sparse object
	elif computation_type == 'sparse':
		row = [nodes.index(k[0]) for k in dN.keys()]
		col = [nodes.index(k[1]) for k in dN.keys()]
		data = [v for v in dN.values()]
		M = coo_matrix((data, (row,col)), shape=(n,n))
		M = M.tocsr()
		M.eliminate_zeros()

	return M, nodes


def _get_operation_from_string(op):
	""" returns function based on a string operation """
	if op in ['union','U','u']:
		return _fuzzy_union

	elif op in ['intersection','u']:
		return _fuzzy_intersection

	elif op in ['difference','diff','d']:
		return _fuzzy_diff

	elif op in ['addition','add','a']:
		return _fuzzy_add

def _fuzzy_union(a,b):
	return max(a,b)

def _fuzzy_intersection(a,b):
	return min(a,b)

def _fuzzy_diff(a,b):
	return (a - b)

def _fuzzy_add(a,b):
	return (a + b)




