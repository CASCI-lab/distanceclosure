
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.spatial.distance import pdist, cdist, squareform
from itertools import combinations
from scipy.sparse.csgraph import shortest_path

d = np.array([
		[0.,.11111111,9.,np.inf],
		[.11111111,0.,0.25,np.inf],
		[9.,0.25,0.,0.66666667],
		[np.inf,np.inf,0.66666667,0.],
], dtype=float)


ds = sp.csr_matrix(d)

apsp = shortest_path(d, method='auto', directed=False)

print d
print apsp