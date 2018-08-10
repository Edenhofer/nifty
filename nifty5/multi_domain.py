# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import absolute_import, division, print_function

from .compat import *
from .domain_tuple import DomainTuple
from .utilities import frozendict


class MultiDomain(object):
    """A tuple of domains corresponding to a direct sum.

    This class is the domain of the direct sum of fields living
    over (possibly different) domains. To make an instance
    of this class, call `MultiDomain.make(inp)`.
    """
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
        """Creates a MultiDomain object from a dictionary of names and domains

        Parameters
        ----------
        inp : MultiDomain or dict{name: DomainTuple}
            The already built MultiDomain or a dictionary of DomainTuples

        Returns
        ------
        A MultiDomain with the input Domains as domains
        """
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
        return self.items() == x.items()

    def __ne__(self, x):
        return not self.__eq__(x)

    def __str__(self):
        res = "MultiDomain:\n"
        for key, dom in zip(self._keys, self._domains):
            res += key+": "+str(dom)+"\n"
        return res

    @staticmethod
    def union(inp):
        inp = set(inp)
        if len(inp) == 1:  # all domains are identical
            return inp.pop()
        res = {}
        for dom in inp:
            for key, subdom in zip(dom._keys, dom._domains):
                if key in res:
                    if res[key] != subdom:
                        raise ValueError("domain mismatch")
                else:
                    res[key] = subdom
        return MultiDomain.make(res)
