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

from distanceclosure.backbone import  distance_backbone, metric_backbone, ultrametric_backbone
from distanceclosure.closure import distance_closure
from distanceclosure.distance import pairwise_proximity
from distanceclosure.utils import prox2dist, dist2prox, dict2matrix, matrix2dict, dict2sparse, from_networkx_to_dijkstra_format, s_values, b_values