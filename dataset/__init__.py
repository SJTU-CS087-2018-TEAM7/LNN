from .LogicSyn import LogicSyn


def as_dataset(data_name, initialized=True):
    data_name = data_name.lower()
    if data_name == 'logicsyn':
        return LogicSyn(initialized=initialized)
    else:
        raise ValueError