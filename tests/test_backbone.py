from distanceclosure.utils import prox2dist, dict2matrix
from distanceclosure.backbone import backbone
import numpy as np
import networkx as nx
#
# Test
#
# Proximity Network
P = np.array([
             [1., .9, .1, 0.],
             [.9, 1., .8, 0.],
             [.1, .8, 1., 1.],
             [0., 0., 1., 1.]], dtype=float)
D = np.vectorize(prox2dist)(P)



def test_drastic_backbone():
    """ Test Closure: Dijkstra vs Dense ultrametric comparison """
    
    G = nx.from_numpy_array(D)
    G.add_weighted_edges_from([(2, 3, 0.0)])
    
    B, s = backbone(G, distortion=True, kind='drastic')
    
    print(G.edges(data=True))
    print(B.edges(data=True))
    print(s)
    
    #return True


test_drastic_backbone()