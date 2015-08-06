def init_logging(level):
    import logging

    logging.basicConfig(level=level, handlers=[], style='{')


from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
