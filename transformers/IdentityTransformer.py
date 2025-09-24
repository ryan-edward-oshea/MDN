from ._CustomTransformer import _CustomTransformer


class IdentityTransformer(_CustomTransformer):
    ''' No transformation '''

    def _fit(self, X, *args, **kwargs):
        # no fitting required, but present for consistency
        return self

    def _transform(self, X, *args, **kwargs):         return X

    def _inverse_transform(self, X, *args, **kwargs): return X

    @staticmethod
    def config_info(*args, **kwargs):
        return "IdentityTransformer"
