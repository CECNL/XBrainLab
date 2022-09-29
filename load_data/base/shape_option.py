from enum import Enum
from itertools import permutations 

class RawShapeOtion(Enum):
    CH = 'channel'
    TIME = 'time'

class EpochShapeOtion(Enum):
    EPOCH = 'epoch'
    CH = 'channel'
    TIME = 'time'

def generate_perm(OPTION):
    shape_option_perm = list(permutations(OPTION))
    shape_option_list = [' x '.join([str(i.value) for i in op]) for op in shape_option_perm]
    return shape_option_perm, shape_option_list
