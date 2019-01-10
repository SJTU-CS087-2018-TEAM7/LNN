from .LogicSyn import LogicSyn
from .ml1m import ml1m


def as_dataset(data_name, initialized=True):
    data_name = data_name.lower()
    if data_name == 'logicsyn':
        return LogicSyn(initialized=initialized)
    elif data_name == 'ml1m':
        return ml1m(initialized=initialized)
    else:
        raise ValueError