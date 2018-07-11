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

from numpy import inf, isnan

from ..compat import *
from ..minimization.energy import Energy
from ..operators.sandwich_operator import SandwichOperator
from ..sugar import log, makeOp


class PoissonianEnergy(Energy):
    def __init__(self, lamb, d):
        """
        lamb: Model object

        value = 0.5 * s.vdot(s), i.e. a log-Gauss distribution with unit
        covariance
        """
        super(PoissonianEnergy, self).__init__(lamb.position)
        self._lamb = lamb
        self._d = d

        lamb_val = self._lamb.value

        self._value = lamb_val.sum() - d.vdot(log(lamb_val))
        if isnan(self._value):
            self._value = inf
        self._gradient = self._lamb.jacobian.adjoint_times(1 - d/lamb_val)

        metric = makeOp(1./lamb_val)
        self._metric = SandwichOperator.make(self._lamb.jacobian, metric)

    def at(self, position):
        return self.__class__(self._lamb.at(position), self._d)

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient

    @property
    def metric(self):
        return self._metric
