def set_seed(seed=None):
    import torch
    import random
    import numpy as np
    from numpy.random import MT19937
    from numpy.random import RandomState, SeedSequence
    

    if seed == None:
        seed = torch.seed()

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rs = RandomState(MT19937(SeedSequence(seed)))
    np.random.set_state(rs.get_state())

    return seed

def get_random_state():
    import torch
    import random
    import numpy as np

    return torch.get_rng_state(), random.getstate(), np.random.get_state()

def set_random_state(state):
    import torch
    import random
    import numpy as np
    
    torch_state, random_state, np_state = state

    torch.torch.set_rng_state(torch_state)
    random.setstate(random_state)
    np.random.set_state(np_state)