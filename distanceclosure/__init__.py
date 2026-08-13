__package__ = 'distanceclosure'
__title__ = "Distance Closure"
__description__ = "Distance Closure on Complex Networks"

__author__ = """\n""".join([
    'Rion Brattig Correia <rionbr@gmail.com>',
    'Luis M. Rocha <rocha@binghamton.edu>',
    'Felipe Xavier Costa <fcosta@binghamton.edu>'
])

__copyright__ = u'2024, Correia, R. B., Costa, F.X., Rocha, L. M.'
__version__ = '0.6.1'

from distanceclosure.backbone import *
from distanceclosure.dijkstra import *
import distanceclosure.utils as dutils


def distance_closure(D, weight='weight', kind='metric', only_reweight=False, cutoff=None, verbose=False, *args, **kwargs):
    """
    Computes the transitive closure (All-Pairs-Shortest-Paths; APSP)
    using different shortest path measures on the distance graph
    (adjacency matrix) with values in the ``[0,inf]`` interval.

    .. math::

        c_{ij} = min_{k}( td_norm ( a_{ik} , b_{kj} ) )

    Parameters
    ----------
    D : NetworkX.Graph
        The Distance graph.

    kind : string
        Type of closure to compute: ``metric`` or ``ultrametric``.

    weight : string
        Edge property containing distance values. Defaults to `weight`.

    cutoff : int, optional (default=None)
        Maximum number of nodes in the path. If None, compute the entire closure as if the cutoff is the number of nodes.
    
    only_reweight : bool
        Only include new distance closure values for edges in the original graph.
    
    Verbose :bool
        Prints statements as it computes.

    Returns
    --------
    C : NetworkX.Graph
        The distance closure graph. Note this may be a fully connected graph.

    Examples
    --------
    >>> distance_closure(D, weight='weight', kind='metric', only_reweight=True)

    - Metric: :math:`(min,+)`
    - Ultrametric: :math:`(min,max)` -- also known as maximum flow.

    .. math::

            [ 1 + \\sum_{i=2}^{n-1} log k(v_i) ]^{-1}
    """
    dutils._check_for_kind(kind)

    G = D.copy()

    if kind == 'metric':
        disjunction = sum
    elif kind == 'ultrametric':
        disjunction = max

    edges_seen = set()
    i = 1
    total = G.number_of_nodes()
    # APSP
    for u, lengths in all_pairs_dijkstra_path_length(G, weight=weight, disjunction=disjunction, cutoff=cutoff):
        if verbose:
            per = i / total
            print(f"Distance Closure : dijkstra : {kind:s} : {i:d} of {total:d} ({per:.2%})")
        for v, length in lengths.items():

            if (u, v) in edges_seen or u == v:
                continue
            else:
                edges_seen.add((u, v))
                kind_distance = f'{kind:s}_distance'
                is_kind = f'is_{kind:s}'
                if not G.has_edge(u, v):
                    if not only_reweight:
                        G.add_edge(u, v, **{weight: float('inf'), kind_distance: length})
                else:
                    G[u][v][kind_distance] = length
                    G[u][v][is_kind] = True if (length == G[u][v][weight]) else False
        i += 1

    return G
