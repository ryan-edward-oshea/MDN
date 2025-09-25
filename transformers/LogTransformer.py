import numpy as np

from ._CustomTransformer import _CustomTransformer


class LogTransformer(_CustomTransformer):
    ''' Transform into log domain '''

    def __init__(self, clip=False):
        self.clip = clip  # store parameter for later
        super().__init__()

    def _fit(self, X, *args, **kwargs):
        # no fitting required, but present for consistency
        return self

    def _transform(self, X, *args, **kwargs):         return np.log(X)

    def _inverse_transform(self, X, *args, **kwargs): return np.exp(X)

    @staticmethod
    def config_info(*args, **kwargs):
        return "LogTransformer"
