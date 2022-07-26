import collections
from typing import Any, Dict

import flax

InfoDict = Dict[str, float]
PRNGKey = Any
Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])
    
Params = flax.core.FrozenDict[str, Any]
