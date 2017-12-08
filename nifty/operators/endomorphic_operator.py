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
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

import abc
from .linear_operator import LinearOperator


class EndomorphicOperator(LinearOperator):
    """ NIFTY class for endomorphic operators.

    The  NIFTY EndomorphicOperator class is a class derived from the
    LinearOperator. By definition, domain and target are the same in
    EndomorphicOperator.

    Attributes
    ----------
    domain : DomainTuple
        The domain on which the Operator's input Field lives.
    target : DomainTuple
        The domain in which the outcome of the operator lives. As the Operator
        is endomorphic this is the same as its domain.
    unitary : boolean
        Indicates whether the Operator is unitary or not.
    self_adjoint : boolean
        Indicates whether the operator is self_adjoint or not.
    """

    def inverse_times(self, x):
        if self.self_adjoint and self.unitary:
            return self.times(x)
        else:
            return super(EndomorphicOperator, self).inverse_times(x)

    def adjoint_times(self, x):
        if self.self_adjoint:
            return self.times(x)
        else:
            return super(EndomorphicOperator, self).adjoint_times(x)

    def adjoint_inverse_times(self, x):
        if self.self_adjoint:
            return self.inverse_times(x)
        else:
            return super(EndomorphicOperator, self).adjoint_inverse_times(x)

    @property
    def target(self):
        return self.domain

    @abc.abstractproperty
    def self_adjoint(self):
        """ States whether the Operator is self_adjoint or not."""
        raise NotImplementedError
