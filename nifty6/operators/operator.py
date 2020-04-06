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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..utilities import NiftyMeta, indent


class Operator(metaclass=NiftyMeta):
    """Transforms values defined on one domain into values defined on another
    domain, and can also provide the Jacobian.
    """

    @property
    def domain(self):
        """The domain on which the Operator's input Field is defined.

        Returns
        -------
        domain : DomainTuple or MultiDomain
        """
        return self._domain

    @property
    def target(self):
        """The domain on which the Operator's output Field is defined.

        Returns
        -------
        target : DomainTuple or MultiDomain
        """
        return self._target

    @property
    def val(self):
        """The numerical value associated with this object
        For "pure" operators this is `None`. For Field-like objects this
        is a `numpy.ndarray` or a dictionary of `numpy.ndarray`s mathcing the
        object's `target`.

        Returns
        -------
        None or numpy.ndarray or dictionary of np.ndarrays : the numerical value
        """
        return None

    @property
    def jac(self):
        """The Jacobian associated with this object
        For "pure" operators this is `None`. For Field-like objects this
        can be `None` (in which case the object is a constant), or it can be a
        `LinearOperator` with `domain` and `target` matching the object's.

        Returns
        -------
        None or LinearOperator : the Jacobian

        Notes
        -----
        if `value` is None, this must be `None` as well!
        """
        return None

    @property
    def want_metric(self):
        """Whether a metric should be computed for the full expression.
        This is `False` whenever `jac` is `None`. In other cases it signals
        that operators processing this object should compute the metric.

        Returns
        -------
        bool : whether the metric should be computed
        """
        return False

    @property
    def metric(self):
        """The metric associated with the object.
        This is `None`, except when all the following conditions hold:
        - `want_metric` is `True`
        - `target` is the scalar domain
        - the operator chain contained an operator which could compute the
          metric

        Returns
        -------
        None or LinearOperator : the metric
        """
        return None

    @staticmethod
    def _check_domain_equality(dom_op, dom_field):
        if dom_op != dom_field:
            s = "The operator's and field's domains don't match."
            from ..domain_tuple import DomainTuple
            from ..multi_domain import MultiDomain
            if not isinstance(dom_op, (DomainTuple, MultiDomain,)):
                s += " Your operator's domain is neither a `DomainTuple`" \
                     " nor a `MultiDomain`."
            raise ValueError(s)

    def scale(self, factor):
        if factor == 1:
            return self
        from .scaling_operator import ScalingOperator
        return ScalingOperator(self.target, factor)(self)

    def conjugate(self):
        from .simple_linear_operators import ConjugationOperator
        return ConjugationOperator(self.target)(self)

    def sum(self, spaces=None):
        from .contraction_operator import ContractionOperator
        return ContractionOperator(self.target, spaces)(self)

    def vdot(self, other):
        from ..sugar import makeOp
        if not isinstance(other, Operator):
            raise TypeError
        if other.jac is None:
            res = self.conjugate()*other
        else:
            res = makeOp(other) @ self.conjugate()
        return res.sum()

    @property
    def real(self):
        from .simple_linear_operators import Realizer
        return Realizer(self.target)(self)

    def __neg__(self):
        return self.scale(-1)

    def __matmul__(self, x):
        if not isinstance(x, Operator):
            return NotImplemented
        return _OpChain.make((self, x))

    def __rmatmul__(self, x):
        if not isinstance(x, Operator):
            return NotImplemented
        return _OpChain.make((x, self))

    def partial_insert(self, x):
        from ..multi_domain import MultiDomain
        if not isinstance(x, Operator):
            raise TypeError
        if not isinstance(self.domain, MultiDomain):
            raise TypeError
        if not isinstance(x.target, MultiDomain):
            raise TypeError
        bigdom = MultiDomain.union([self.domain, x.target])
        k1, k2 = set(self.domain.keys()), set(x.target.keys())
        le, ri = k2 - k1, k1 - k2
        leop, riop = self, x
        if len(ri) > 0:
            riop = riop + self.identity_operator(
                MultiDomain.make({kk: bigdom[kk]
                                  for kk in ri}))
        if len(le) > 0:
            leop = leop + self.identity_operator(
                MultiDomain.make({kk: bigdom[kk]
                                  for kk in le}))
        return leop @ riop

    @staticmethod
    def identity_operator(dom):
        from .block_diagonal_operator import BlockDiagonalOperator
        from .scaling_operator import ScalingOperator
        idops = {kk: ScalingOperator(dd, 1.) for kk, dd in dom.items()}
        return BlockDiagonalOperator(dom, idops)

    def __mul__(self, x):
        if isinstance(x, Operator):
            return _OpProd(self, x)
        if np.isscalar(x):
            return self.scale(x)
        return NotImplemented

    def __rmul__(self, x):
        return self.__mul__(x)

    def __add__(self, x):
        if not isinstance(x, Operator):
            return NotImplemented
        return _OpSum(self, x)

    def __sub__(self, x):
        if not isinstance(x, Operator):
            return NotImplemented
        return _OpSum(self, -x)

    def __pow__(self, power):
        if not (np.isscalar(power) or power.jac is None):
            return NotImplemented
        return self.ptw("power", power)

    def clip(self, a_min=None, a_max=None):
        if a_min is None and a_max is None:
            return self
        if not (a_min is None or np.isscalar(a_min) or a_min.jac is None):
            return NotImplemented
        if not (a_max is None or np.isscalar(a_max) or a_max.jac is None):
            return NotImplemented
        return self.ptw("clip", a_min, a_max)

    def apply(self, x):
        """Applies the operator to a Field or MultiField.

        Parameters
        ----------
        x : Field or MultiField
            Input on which the operator shall act. Needs to be defined on
            :attr:`domain`.
        """
        raise NotImplementedError

    def force(self, x):
        """Extract subset of domain of x according to `self.domain` and apply
        operator."""
        return self.apply(x.extract(self.domain))

    def _check_input(self, x):
        from .scaling_operator import ScalingOperator
        if not (isinstance(x, Operator) and x.val is not None):
            raise TypeError
        if x.jac is not None:
            if not isinstance(x.jac, ScalingOperator):
                raise ValueError
            if x.jac._factor != 1:
                raise ValueError
        self._check_domain_equality(self._domain, x.domain)

    def __call__(self, x):
        if not isinstance(x, Operator):
            raise TypeError
        if x.jac is not None:
            return self.apply(x.trivial_jac()).prepend_jac(x.jac)
        elif x.val is not None:
            return self.apply(x)
        return self @ x

    def ducktape(self, name):
        from .simple_linear_operators import ducktape
        return self @ ducktape(self, None, name)

    def ducktape_left(self, name):
        from .simple_linear_operators import ducktape
        return ducktape(None, self, name) @ self

    def __repr__(self):
        return self.__class__.__name__

    def simplify_for_constant_input(self, c_inp):
        if c_inp is None:
            return None, self
        if c_inp.domain == self.domain:
            op = _ConstantOperator(self.domain, self(c_inp))
            return op(c_inp), op
        return self._simplify_for_constant_input_nontrivial(c_inp)

    def _simplify_for_constant_input_nontrivial(self, c_inp):
        return None, self

    def ptw(self, op, *args, **kwargs):
        return _OpChain.make((_FunctionApplier(self.target, op, *args, **kwargs), self))


class _ConstCollector(object):
    def __init__(self):
        self._const = None
        self._nc = set()

    def mult(self, const, fulldom):
        if const is None:
            self._nc |= set(fulldom)
        else:
            self._nc |= set(fulldom) - set(const)
            if self._const is None:
                from ..multi_field import MultiField
                self._const = MultiField.from_dict(
                    {key: const[key] for key in const if key not in self._nc})
            else:
                from ..multi_field import MultiField
                self._const = MultiField.from_dict(
                    {key: self._const[key]*const[key]
                     for key in const if key not in self._nc})

    def add(self, const, fulldom):
        if const is None:
            self._nc |= set(fulldom.keys())
        else:
            from ..multi_field import MultiField
            self._nc |= set(fulldom.keys()) - set(const.keys())
            if self._const is None:
                self._const = MultiField.from_dict(
                    {key: const[key]
                     for key in const.keys() if key not in self._nc})
            else:
                self._const = self._const.unite(const)
                self._const = MultiField.from_dict(
                    {key: self._const[key]
                     for key in self._const if key not in self._nc})

    @property
    def constfield(self):
        return self._const


class _ConstantOperator(Operator):
    def __init__(self, dom, output):
        from ..sugar import makeDomain
        self._domain = makeDomain(dom)
        self._target = output.domain
        self._output = output

    def apply(self, x):
        from .simple_linear_operators import NullOperator
        self._check_input(x)
        if x.jac is not None:
            return x.new(self._output, NullOperator(self._domain, self._target))
        return self._output

    def __repr__(self):
        return 'ConstantOperator <- {}'.format(self.domain.keys())


class _FunctionApplier(Operator):
    def __init__(self, domain, funcname, *args, **kwargs):
        from ..sugar import makeDomain
        self._domain = self._target = makeDomain(domain)
        self._funcname = funcname
        self._args = args
        self._kwargs = kwargs

    def apply(self, x):
        self._check_input(x)
        return x.ptw(self._funcname, *self._args, **self._kwargs)


class _CombinedOperator(Operator):
    def __init__(self, ops, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError
        self._ops = tuple(ops)

    @classmethod
    def unpack(cls, ops, res):
        for op in ops:
            if isinstance(op, cls):
                res = cls.unpack(op._ops, res)
            else:
                res = res + [op]
        return res

    @classmethod
    def make(cls, ops):
        res = cls.unpack(ops, [])
        if len(res) == 1:
            return res[0]
        return cls(res, _callingfrommake=True)


class _OpChain(_CombinedOperator):
    def __init__(self, ops, _callingfrommake=False):
        super(_OpChain, self).__init__(ops, _callingfrommake)
        self._domain = self._ops[-1].domain
        self._target = self._ops[0].target
        for i in range(1, len(self._ops)):
            if self._ops[i-1].domain != self._ops[i].target:
                raise ValueError("domain mismatch")

    def apply(self, x):
        self._check_input(x)
        for op in reversed(self._ops):
            x = op(x)
        return x

    def _simplify_for_constant_input_nontrivial(self, c_inp):
        from ..multi_domain import MultiDomain
        if not isinstance(self._domain, MultiDomain):
            return None, self

        newop = None
        for op in reversed(self._ops):
            c_inp, t_op = op.simplify_for_constant_input(c_inp)
            newop = t_op if newop is None else op(newop)
        return c_inp, newop

    def __repr__(self):
        subs = "\n".join(sub.__repr__() for sub in self._ops)
        return "_OpChain:\n" + indent(subs)


class _OpProd(Operator):
    def __init__(self, op1, op2):
        from ..sugar import domain_union
        self._domain = domain_union((op1.domain, op2.domain))
        self._target = op1.target
        if op1.target != op2.target:
            raise ValueError("target mismatch")
        self._op1 = op1
        self._op2 = op2

    def apply(self, x):
        from ..linearization import Linearization
        from ..sugar import makeOp
        self._check_input(x)
        lin = x.jac is not None
        wm = x.want_metric if lin else False
        x = x.val if lin else x
        v1 = x.extract(self._op1.domain)
        v2 = x.extract(self._op2.domain)
        if not lin:
            return self._op1(v1) * self._op2(v2)
        lin1 = self._op1(Linearization.make_var(v1, wm))
        lin2 = self._op2(Linearization.make_var(v2, wm))
        jac = (makeOp(lin1._val)(lin2._jac))._myadd(makeOp(lin2._val)(lin1._jac), False)
        return lin1.new(lin1._val*lin2._val, jac)

    def _simplify_for_constant_input_nontrivial(self, c_inp):
        f1, o1 = self._op1.simplify_for_constant_input(
            c_inp.extract_part(self._op1.domain))
        f2, o2 = self._op2.simplify_for_constant_input(
            c_inp.extract_part(self._op2.domain))

        from ..multi_domain import MultiDomain
        if not isinstance(self._target, MultiDomain):
            return None, _OpProd(o1, o2)

        cc = _ConstCollector()
        cc.mult(f1, o1.target)
        cc.mult(f2, o2.target)
        return cc.constfield, _OpProd(o1, o2)

    def __repr__(self):
        subs = "\n".join(sub.__repr__() for sub in (self._op1, self._op2))
        return "_OpProd:\n"+indent(subs)


class _OpSum(Operator):
    def __init__(self, op1, op2):
        from ..sugar import domain_union
        self._domain = domain_union((op1.domain, op2.domain))
        self._target = domain_union((op1.target, op2.target))
        self._op1 = op1
        self._op2 = op2

    def apply(self, x):
        from ..linearization import Linearization
        self._check_input(x)
        if x.jac is None:
            v1 = x.extract(self._op1.domain)
            v2 = x.extract(self._op2.domain)
            return self._op1(v1).unite(self._op2(v2))
        v1 = x.val.extract(self._op1.domain)
        v2 = x.val.extract(self._op2.domain)
        wm = x.want_metric
        lin1 = self._op1(Linearization.make_var(v1, wm))
        lin2 = self._op2(Linearization.make_var(v2, wm))
        op = lin1._jac._myadd(lin2._jac, False)
        res = lin1.new(lin1._val.unite(lin2._val), op)
        if lin1._metric is not None and lin2._metric is not None:
            res = res.add_metric(lin1._metric._myadd(lin2._metric, False))
        return res

    def _simplify_for_constant_input_nontrivial(self, c_inp):
        f1, o1 = self._op1.simplify_for_constant_input(
            c_inp.extract_part(self._op1.domain))
        f2, o2 = self._op2.simplify_for_constant_input(
            c_inp.extract_part(self._op2.domain))

        from ..multi_domain import MultiDomain
        if not isinstance(self._target, MultiDomain):
            return None, _OpSum(o1, o2)

        cc = _ConstCollector()
        cc.add(f1, o1.target)
        cc.add(f2, o2.target)
        return cc.constfield, _OpSum(o1, o2)

    def __repr__(self):
        subs = "\n".join(sub.__repr__() for sub in (self._op1, self._op2))
        return "_OpSum:\n"+indent(subs)
