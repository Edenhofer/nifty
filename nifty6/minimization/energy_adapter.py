# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from ..linearization import Linearization
from ..minimization.energy import Energy


class EnergyAdapter(Energy):
    """Helper class which provides the traditional Nifty Energy interface to
    Nifty operators with a scalar target domain.

    Parameters
    -----------
    position: Field or MultiField
        The position where the minimization process is started.
    op: EnergyOperator
        The expression computing the energy from the input data.
    constants: list of strings
        The component names of the operator's input domain which are assumed
        to be constant during the minimization process.
        If the operator's input domain is not a MultiField, this must be empty.
        Default: [].
    want_metric: bool
        If True, the class will provide a `metric` property. This should only
        be enabled if it is required, because it will most likely consume
        additional resources. Default: False.
    """

    def __init__(self, position, op, constants=[], want_metric=False):
        super(EnergyAdapter, self).__init__(position)
        self._op = op
        self._constants = constants
        self._want_metric = want_metric
        lin = Linearization.make_partial_var(position, constants, want_metric)
        tmp = self._op(lin)
        self._val = tmp.val.val[()]
        self._grad = tmp.gradient
        self._metric = tmp._metric

    def at(self, position):
        return EnergyAdapter(position, self._op, self._constants,
                             self._want_metric)

    @property
    def value(self):
        return self._val

    @property
    def gradient(self):
        return self._grad

    @property
    def metric(self):
        return self._metric

    def apply_metric(self, x):
        return self._metric(x)
