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

import abc
from future.utils import with_metaclass


class LineSearch(with_metaclass(abc.ABCMeta,
                                with_metaclass(abc.ABCMeta,
                                               type('NewBase',
                                                    (object,), {})))):
    """Class for determining the optimal step size along some descent direction.

    Initialize the line search procedure which can be used by a specific line
    search method. It finds the step size in a specific direction in the
    minimization process.
    """

    def __init__(self, preferred_initial_step_size=None):
        self.preferred_initial_step_size = preferred_initial_step_size

    @abc.abstractmethod
    def perform_line_search(self, energy, pk, f_k_minus_1=None):
        raise NotImplementedError
