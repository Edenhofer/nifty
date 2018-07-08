from __future__ import absolute_import, division, print_function
from ..compat import *
from ..domain_tuple import DomainTuple
from ..utilities import frozendict


class MultiDomain(object):
    _domainCache = {}

    def __init__(self, dict, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError(
                'To create a MultiDomain call `MultiDomain.make()`.')
        self._keys = tuple(sorted(dict.keys()))
        self._domains = tuple(dict[key] for key in self._keys)
        self._idx = frozendict({key: i for i, key in enumerate(self._keys)})

    @staticmethod
    def make(inp):
        if isinstance(inp, MultiDomain):
            return inp
        if not isinstance(inp, dict):
            raise TypeError("dict expected")
        tmp = {}
        for key, value in inp.items():
            if not isinstance(key, str):
                raise TypeError("keys must be strings")
            tmp[key] = DomainTuple.make(value)
        tmp = frozendict(tmp)
        obj = MultiDomain._domainCache.get(tmp)
        if obj is not None:
            return obj
        obj = MultiDomain(tmp, _callingfrommake=True)
        MultiDomain._domainCache[tmp] = obj
        return obj

    def keys(self):
        return self._keys

    def domains(self):
        return self._domains

    @property
    def idx(self):
        return self._idx

    def items(self):
        return zip(self._keys, self._domains)

    def __getitem__(self, key):
        return self._domains[self._idx[key]]

    def __len__(self):
        return len(self._keys)

    def __hash__(self):
        return self._keys.__hash__() ^ self._domains.__hash__()

    def __eq__(self, x):
        if self is x:
            return True
        return self is MultiDomain.make(x)

    def __ne__(self, x):
        return not self.__eq__(x)

    def __str__(self):
        res = "MultiDomain:\n"
        for key, dom  in zip(self._keys, self._domains):
            res += key+": "+str(dom)+"\n"
        return res
