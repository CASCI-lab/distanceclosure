__package__ = 'distanceclosure'
__title__ = "Distance Closure"
__description__ = "Distance Closure on Complex Networks"

__author__ = """\n""".join([
    'Rion Brattig Correia <rionbr@gmail.com>',
    'Luis M. Rocha <rocha@binghamton.edu>',
    'Felipe Xavier Costa <fcosta@binghamton.edu>'
])

__copyright__ = u'2020, Correia, R. B., Rocha, L. M.'

__version__ = '0.5'

from distanceclosure.backbone import *
from distanceclosure.dijkstra import *
from distanceclosure.closure import *
import distanceclosure.utils
