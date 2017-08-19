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
import numpy as np

from .minimizer import Minimizer
from .line_searching import LineSearchStrongWolfe


class DescentMinimizer(Minimizer):
    """ A base class used by gradient methods to find a local minimum.

    Descent minimization methods are used to find a local minimum of a scalar
    function by following a descent direction. This class implements the
    minimization procedure once a descent direction is known. The descent
    direction has to be implemented separately.

    Parameters
    ----------
    line_searcher : callable *optional*
        Function which infers the step size in the descent direction
        (default : LineSearchStrongWolfe()).
    callback : callable *optional*
        Function f(energy, iteration_number) supplied by the user to perform
        in-situ analysis at every iteration step. When being called the
        current energy and iteration_number are passed. (default: None)

    Attributes
    ----------
    line_searcher : LineSearch
        Function which infers the optimal step size for functional minization
        given a descent direction.
    callback : function
        Function f(energy, iteration_number) supplied by the user to perform
        in-situ analysis at every iteration step. When being called the
        current energy and iteration_number are passed.

    Notes
    ------
    The callback function can be used to externally stop the minimization by
    raising a `StopIteration` exception.
    Check `get_descent_direction` of a derived class for information on the
    concrete minization scheme.

    """

    def __init__(self, controller, line_searcher=LineSearchStrongWolfe()):
        super(DescentMinimizer, self).__init__()

        self.line_searcher = line_searcher
        self._controller = controller

    def __call__(self, energy):
        """ Performs the minimization of the provided Energy functional.

        Parameters
        ----------
        energy : Energy object
           Energy object which provides value, gradient and curvature at a
           specific position in parameter space.

        Returns
        -------
        energy : Energy object
            Latest `energy` of the minimization.
        convergence : integer
            Latest convergence level indicating whether the minimization
            has converged or not.

        Note
        ----
        The minimization is stopped if
            * the callback function raises a `StopIteration` exception,
            * a perfectly flat point is reached,
            * according to the line-search the minimum is found,
            * the target convergence level is reached,
            * the iteration limit is reached.

        """

        f_k_minus_1 = None
        controller = self._controller
        status = controller.start(energy)
        if status != controller.CONTINUE:
            return E, status

        while True:
            # check if position is at a flat point
            if energy.gradient_norm == 0:
                self.logger.info("Reached perfectly flat point. Stopping.")
                return energy, controller.CONVERGED

            # current position is encoded in energy object
            descent_direction = self.get_descent_direction(energy)
            # compute the step length, which minimizes energy.value along the
            # search direction
            try:
                new_energy = \
                    self.line_searcher.perform_line_search(
                                                   energy=energy,
                                                   pk=descent_direction,
                                                   f_k_minus_1=f_k_minus_1)
            except RuntimeError:
                self.logger.warn(
                        "Stopping because of RuntimeError in line-search")
                return energy, controller.ERROR

            f_k_minus_1 = energy.value
            # check if new energy value is bigger than old energy value
            if (new_energy.value - energy.value) > 0:
                self.logger.info("Line search algorithm returned a new energy "
                                 "that was larger than the old one. Stopping.")
                return energy, controller.ERROR

            energy = new_energy
            status = self._controller.check(energy)
            if status != controller.CONTINUE:
                return energy, status


    @abc.abstractmethod
    def get_descent_direction(self, energy):
        raise NotImplementedError
