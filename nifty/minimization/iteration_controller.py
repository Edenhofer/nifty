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
from nifty.nifty_meta import NiftyMeta

import numpy as np

from keepers import Loggable

class IterationController(Loggable, object):

    __metaclass__ = NiftyMeta

    CONVERGED, CONTINUE, ERROR = range(3)

    @abc.abstractmethod
    def start(self, energy):
        """
        Parameters
        ----------
        energy : Energy object
           Energy object at the start of the iteration

        Returns
        -------
        status : integer status, can be CONVERGED, CONTINUE or ERROR
        """

        raise NotImplementedError

    @abc.abstractmethod
    def check(self, energy):
        """
        Parameters
        ----------
        energy : Energy object
           Energy object at the start of the iteration

        Returns
        -------
        status : integer status, can be CONVERGED, CONTINUE or ERROR
        """

        raise NotImplementedError
