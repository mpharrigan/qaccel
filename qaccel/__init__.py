from .param import Param
from .run import Run


def init_logging(level):
    import logging

    logging.basicConfig(level=level, handlers=[], style='{')
