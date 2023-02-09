__package__ = 'distanceclosure'
__title__ = "Distance Closure"
__description__ = "Distance Closure on Complex Networks"

__author__ = """\n""".join([
    'Rion Brattig Correia <rionbr@gmail.com>',
    'Felipe Xavier Costa <fcosta@binghamton.com>',
    'Luis M. Rocha <rocha@binghamton.edu>'
])

__copyright__ = u'2023, Correia, R. B., Costa, F. X., Rocha, L. M.'

__version__ = '0.4.1'

from distanceclosure.backbone import *
from distanceclosure.dijkstra import *
from distanceclosure.closure import *
import distanceclosure.utils
