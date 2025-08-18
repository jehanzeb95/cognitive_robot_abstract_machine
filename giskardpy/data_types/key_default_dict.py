from collections import defaultdict


class KeyDefaultDict(defaultdict):
    """
    A default dict where the key is passed as parameter to the factory function.
    """

    def __missing__(self, key, cache=True):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            if cache:
                ret = self[key] = self.default_factory(key)
                return ret
            else:
                return self.default_factory(key)