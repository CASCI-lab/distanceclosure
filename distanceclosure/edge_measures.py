"""
Edge Measures from Distance Closure
"""

from distanceclosure.backbone import metric_backbone, ultrametric_backbone
from distanceclosure.closure import distance_closure
import networkx as nx
import numpy as np
import multiprocessing as mp


__name__ = 'distanceclosure'
__author__ = """\n""".join(['Felipe Xavier Costa <fcosta@binghamton.com>',
                            'Bernardo Pereira <mbernardogp@gmail.com>'])

__all__ = [
    "edge_distortion",
    "dombi_synthesis",
    "below_average_ratio"
]


def edge_distortion(D, weight='weight', kind='metric', self_loops=False, cutoff=None, verbose=False, *args, **kwargs):
    
    if kind == 'metric':
        G, svals = metric_backbone(D, weight=weight, distortion=True, self_loops=self_loops, cutoff=cutoff, verbose=verbose, *args, **kwargs)
    elif kind == 'ultrametric':
        G, svals = ultrametric_backbone(D, weight=weight, distortion=True, self_loops=self_loops, cutoff=cutoff, verbose=verbose, *args, **kwargs)
    else:
        raise ValueError("Invalid kind. Choose 'metric' or 'ultrametric'.")

    for u, v in G.edges():
        svals[(u, v)] = 1.0
        
    nx.set_edge_attributes(D, values=svals, name=f'{kind}_distortion')
    
    return D


def below_average_ratio(D, weight='weight', kind='metric', self_loops=False, cutoff=None, verbose=False, *args, **kwargs):
    """
    Computes below-average ratios for each edge with infinite distance, thus not existing in the original distance graph.
    The formal definition is as follow:

    .. math::
        b_{ij} = <d_{ik}> / d_{ij}^m

        b_{ji} = <d_{jk}> / d_{ij}^m

    which is the average distance of all edges that leaves from node `x_i`, divided by its triangular distance.
    Note that `b_{ij}` can be different from `b_{ji}`.

    Parameters
    ----------
    D (networkx.Graph): The original distance graph.
    weight (string): Edge attribute containing distance values. Defaults to 'weight'.
    kind (string): The type of distance closure to use. Defaults to 'metric'.
    self_loops (bool): Whether to include self-loops in the computation. Defaults to False.
    cutoff (float): The maximum distance to consider. Defaults to None.
    verbose (bool): Whether to print progress information. Defaults to False.

    Warning
    -------
    This computation takes a while.
    """

    GC = distance_closure(D, weight=weight, kind=kind, only_reweight=False, self_loops=False, cutoff=None, verbose=verbose, *args, **kwargs)

    if GC.is_directed():
        sout = GC.out_degree(weight=weight)
        kout = GC.out_degree()
    else:
        sout = GC.degree(weight=weight)
        kout = GC.degree()

    mean_distance = {k: sout[k] / kout[k] if kout[k] > 0 else 0 for k in GC.nodes()}

    G = nx.DiGraph()
    G.add_nodes_from(GC.nodes(data=True))

    for u, v, d in GC.edges(data=f'{kind:s}_distance'):
        if not D.has_edge(u, v):
            G.add_weighted_edge(u, v, ratio=mean_distance[u]/d)
            if not GC.is_directed():
                G.add_weighted_edge(v, u, ratio=mean_distance[v]/d)

    return G


def dombi_synthesis(D, prox_weight='weight', L=1e-4, R=1e3, ntrials=50, ncpu=1):

    G = D.copy()
    nx.set_edge_attributes(G, name='distance', values={(u, v): (1/w - 1) for u, v, w in G.edges(data=prox_weight)})
    U = ultrametric_backbone(G, weight='distance', distortion=False)
    
    ultrametric_edges = list(U.edges())
    non_ultrametric_edges = list(set(G.edges()) - set(U.edges()))

    nr_non_ultrametric_edges = len(non_ultrametric_edges)

    #Parallelized computation of the lambdas
    inputs = zip([G for _ in range(nr_non_ultrametric_edges)],
                 [edge for edge in non_ultrametric_edges], 
                 [L for _ in range(nr_non_ultrametric_edges)],
                 [R for _ in range(nr_non_ultrametric_edges)],
                 [ntrials for _ in range(nr_non_ultrametric_edges)])

    with mp.Pool(processes=ncpu) as pool:
        non_ultrametric_edges_lambdas = pool.starmap(_get_largest_lambda_with_edge_in_backbone, inputs)
        pool.close()

    #Create dictionary that assigns a lambda to each edge
    edges = non_ultrametric_edges + ultrametric_edges
    lambdas = non_ultrametric_edges_lambdas + [np.inf for _ in range(len(ultrametric_edges))]
    edge_to_lambda = dict(zip(edges, lambdas))

    #Save the Edge:Lambda correspondence as a network with lambdas in the edge attributes
    nx.set_edge_attributes(D, edge_to_lambda, name=f'synthesis_parameter')

    return D


def _get_largest_lambda_with_edge_in_backbone(net, edge, L, R, ntrial):

    res='None'
    for l in np.arange(L, R, 5):
        r=l+5

        l_val, r_val = _is_edge_in_backbone(net, l, edge), _is_edge_in_backbone(net, r, edge)
        #I want to compute the largest l such that the edge is in the backbone, using a binary search with k steps
        if l_val==True and r_val==False:
            for _ in range(ntrial):
                mid = (l+r)/2
                mid_val = _is_edge_in_backbone(net, mid, edge)
                if mid_val == True:
                    l = mid
                else:
                    r = mid
            res = l
            break
        
    if res=='None':
        print('The given initial bounds do not give any information with a initial search of width 5 | L:', L, 'R:', R)
        
    #print(f'Edge {edge} | Lambda:', res)
    return res


def _is_edge_in_backbone(net, l, edge):

    nx.set_edge_attributes(net, name='new_distance', values={(u, v): d**l for u, v, d in net.edges(data='distance')})
    sp = nx.shortest_path(net,  edge[0], edge[1], weight='new_distance')

    if len(sp)==2: 
        return True
    else: 
        return False
