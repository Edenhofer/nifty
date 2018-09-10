from __future__ import absolute_import, division, print_function

import numpy as np

from .compat import *
from .field import Field
from .multi_field import MultiField
from .sugar import makeOp


class Linearization(object):
    def __init__(self, val, jac, metric=None, want_metric=False):
        self._val = val
        self._jac = jac
        if self._val.domain != self._jac.target:
            raise ValueError("domain mismatch")
        self._want_metric = want_metric
        self._metric = metric

    def new(self, val, jac, metric=None):
        return Linearization(val, jac, metric, self._want_metric)

    @property
    def domain(self):
        return self._jac.domain

    @property
    def target(self):
        return self._jac.target

    @property
    def val(self):
        return self._val

    @property
    def jac(self):
        return self._jac

    @property
    def gradient(self):
        """Only available if target is a scalar"""
        return self._jac.adjoint_times(Field.scalar(1.))

    @property
    def want_metric(self):
        return self._want_metric

    @property
    def metric(self):
        """Only available if target is a scalar"""
        return self._metric

    def __getitem__(self, name):
        from .operators.simple_linear_operators import FieldAdapter
        return self.new(self._val[name], FieldAdapter(self.domain, name))

    def __neg__(self):
        return self.new(-self._val, -self._jac,
                        None if self._metric is None else -self._metric)

    def conjugate(self):
        return self.new(
            self._val.conjugate(), self._jac.conjugate(),
            None if self._metric is None else self._metric.conjugate())

    @property
    def real(self):
        return self.new(self._val.real, self._jac.real)

    def _myadd(self, other, neg):
        if isinstance(other, Linearization):
            met = None
            if self._metric is not None and other._metric is not None:
                met = self._metric._myadd(other._metric, neg)
            return self.new(
                self._val.flexible_addsub(other._val, neg),
                self._jac._myadd(other._jac, neg), met)
        if isinstance(other, (int, float, complex, Field, MultiField)):
            if neg:
                return self.new(self._val-other, self._jac, self._metric)
            else:
                return self.new(self._val+other, self._jac, self._metric)

    def __add__(self, other):
        return self._myadd(other, False)

    def __radd__(self, other):
        return self._myadd(other, False)

    def __sub__(self, other):
        return self._myadd(other, True)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __truediv__(self, other):
        if isinstance(other, Linearization):
            return self.__mul__(other.inverse())
        return self.__mul__(1./other)

    def __rtruediv__(self, other):
        return self.inverse().__mul__(other)

    def inverse(self):
        return self.new(1./self._val, makeOp(-1./(self._val**2))(self._jac))

    def __mul__(self, other):
        from .sugar import makeOp
        if isinstance(other, Linearization):
            if self.target != other.target:
                raise ValueError("domain mismatch")
            return self.new(
                self._val*other._val,
                (makeOp(other._val)(self._jac))._myadd(
                 makeOp(self._val)(other._jac), False))
        if np.isscalar(other):
            if other == 1:
                return self
            met = None if self._metric is None else self._metric.scale(other)
            return self.new(self._val*other, self._jac.scale(other), met)
        if isinstance(other, (Field, MultiField)):
            if self.target != other.domain:
                raise ValueError("domain mismatch")
            return self.new(self._val*other, makeOp(other)(self._jac))

    def __rmul__(self, other):
        return self.__mul__(other)

    def vdot(self, other):
        from .operators.simple_linear_operators import VdotOperator
        if isinstance(other, (Field, MultiField)):
            return self.new(
                Field.scalar(self._val.vdot(other)),
                VdotOperator(other)(self._jac))
        return self.new(
            Field.scalar(self._val.vdot(other._val)),
            VdotOperator(self._val)(other._jac) +
            VdotOperator(other._val)(self._jac))

    def sum(self):
        from .operators.simple_linear_operators import SumReductionOperator
        return self.new(
            Field.scalar(self._val.sum()),
            SumReductionOperator(self._jac.target)(self._jac))

    def exp(self):
        tmp = self._val.exp()
        return self.new(tmp, makeOp(tmp)(self._jac))

    def log(self):
        tmp = self._val.log()
        return self.new(tmp, makeOp(1./self._val)(self._jac))

    def tanh(self):
        tmp = self._val.tanh()
        return self.new(tmp, makeOp(1.-tmp**2)(self._jac))

    def positive_tanh(self):
        tmp = self._val.tanh()
        tmp2 = 0.5*(1.+tmp)
        return self.new(tmp2, makeOp(0.5*(1.-tmp**2))(self._jac))

    def add_metric(self, metric):
        return self.new(self._val, self._jac, metric)

    def with_want_metric(self):
        return Linearization(self._val, self._jac, self._metric, True)

    @staticmethod
    def make_var(field, want_metric=False):
        from .operators.scaling_operator import ScalingOperator
        return Linearization(field, ScalingOperator(1., field.domain),
                             want_metric=want_metric)

    @staticmethod
    def make_const(field, want_metric=False):
        from .operators.simple_linear_operators import NullOperator
        return Linearization(field, NullOperator(field.domain, field.domain),
                             want_metric=want_metric)

    @staticmethod
    def make_partial_var(field, constants, want_metric=False):
        from .operators.scaling_operator import ScalingOperator
        from .operators.block_diagonal_operator import BlockDiagonalOperator
        if len(constants) == 0:
            return Linearization.make_var(field, want_metric)
        else:
            ops = [ScalingOperator(0. if key in constants else 1., dom)
                   for key, dom in field.domain.items()]
            bdop = BlockDiagonalOperator(field.domain, tuple(ops))
            return Linearization(field, bdop, want_metric=want_metric)