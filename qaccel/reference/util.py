from pkg_resources import resource_filename


def get_fn(fn):
    return resource_filename('qaccel', 'reference/data/{}'.format(fn))
