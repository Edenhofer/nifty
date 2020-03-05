# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from functools import reduce
from ..utilities import NiftyMeta


class Domain(metaclass=NiftyMeta):
    """The abstract class repesenting a (structured or unstructured) domain.
    """
    def __repr__(self):
        raise NotImplementedError

    def __hash__(self):
        """Returns a hash value for the object.

        Notes
        -----
        Only members that are explicitly added to
        :attr:`._needed_for_hash` will be used for hashing.
        """
        try:
            return self._hash
        except AttributeError:
            v = vars(self)
            self._hash = reduce(lambda x, y: x ^ y, (hash(v[key])
                                for key in self._needed_for_hash), 0)
        return self._hash

    def __eq__(self, x):
        """Checks whether two domains are equal.

        Parameters
        ----------
        x : Domain
            The domain `self` is compared to.

        Returns
        -------
        bool : True iff `self` and x describe the same domain.

        Notes
        -----
        Only members that are explicitly added to
        :attr:`._needed_for_hash` will be used for comparison.

        Subclasses of Domain should not re-define :meth:`__eq__`,
        :meth:`__ne__`, or :meth:`__hash__`; they should instead add their
        relevant attributes' names to :attr:`._needed_for_hash`.
        """
        if self is x:  # shortcut for simple case
            return True
        if not isinstance(x, type(self)):
            return False
        for key in self._needed_for_hash:
            if vars(self)[key] != vars(x)[key]:
                return False
        return True

    def __ne__(self, x):
        """Returns the opposite of :meth:`.__eq__()`"""
        return not self.__eq__(x)

    @property
    def shape(self):
        """tuple of int: number of pixels along each axis

        The shape of the array-like object required to store information
        defined on the domain.
        """
        raise NotImplementedError

    @property
    def size(self):
        """int: total number of pixels.

        Equivalent to the products over all entries in the domain's shape.
        """
        raise NotImplementedError
