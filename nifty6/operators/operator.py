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

from ..field import Field
from ..multi_field import MultiField
from ..utilities import NiftyMeta, indent


class Operator(metaclass=NiftyMeta):
    """Transforms values defined on one domain into values defined on another
    domain, and can also provide the Jacobian.
    """

    VALUE_ONLY = 0
    WITH_JAC = 1
    WITH_METRIC = 2

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
        from ..field import Field
        from ..multi_field import MultiField
        from ..sugar import makeOp
        if isinstance(other, Operator):
            res = self.conjugate()*other
        elif isinstance(other, (Field, MultiField)):
            res = makeOp(other) @ self.conjugate()
        else:
            raise TypeError
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
        if not np.isscalar(power):
            return NotImplemented
        return _OpChain.make((_PowerOp(self.target, power), self))

    def clip(self, min=None, max=None):
        if min is None and max is None:
            return self
        return _OpChain.make((_Clipper(self.target, min, max), self))

    def apply(self, x, difforder):
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
        return self.apply(x.extract(self.domain), 0)

    def _check_input(self, x):
        if not isinstance(x, (Field, MultiField)):
            raise TypeError
        self._check_domain_equality(self._domain, x.domain)

    def __call__(self, x):
        from ..linearization import Linearization
        from ..field import Field
        from ..multi_field import MultiField
        if isinstance(x, Linearization):
            difforder = self.WITH_METRIC if x.want_metric else self.WITH_JAC
            return self.apply(x.val, difforder).prepend_jac(x.jac)
        elif isinstance(x, (Field, MultiField)):
            return self.apply(x, self.VALUE_ONLY)
        raise TypeError('Operator can only consume Field, MultiFields and Linearizations')

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


for f in ["sqrt", "exp", "log", "sin", "cos", "tan", "sinh", "cosh", "tanh",
          "sinc", "sigmoid", "absolute", "one_over", "log10", "log1p", "expm1"]:
    def func(f):
        def func2(self):
            fa = _FunctionApplier(self.target, f)
            return _OpChain.make((fa, self))
        return func2
    setattr(Operator, f, func(f))


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

    def apply(self, x, difforder):
        from ..linearization import Linearization
        from .simple_linear_operators import NullOperator
        self._check_input(x)
        if difforder >= self.WITH_JAC:
            return Linearization(self._output, NullOperator(self._domain, self._target))
        return self._output

    def __repr__(self):
        return 'ConstantOperator <- {}'.format(self.domain.keys())


class _FunctionApplier(Operator):
    def __init__(self, domain, funcname):
        from ..sugar import makeDomain
        self._domain = self._target = makeDomain(domain)
        self._funcname = funcname

    def apply(self, x, difforder):
        self._check_input(x)
        from ..linearization import Linearization
        if difforder >= self.WITH_JAC:
            x = Linearization.make_var(x, difforder == self.WITH_METRIC)
        return getattr(x, self._funcname)()


class _Clipper(Operator):
    def __init__(self, domain, min=None, max=None):
        from ..sugar import makeDomain
        self._domain = self._target = makeDomain(domain)
        self._min = min
        self._max = max

    def apply(self, x, difforder):
        self._check_input(x)
        from ..linearization import Linearization
        if difforder >= self.WITH_JAC:
            x = Linearization.make_var(x, difforder == self.WITH_METRIC)
        return x.clip(self._min, self._max)


class _PowerOp(Operator):
    def __init__(self, domain, power):
        from ..sugar import makeDomain
        self._domain = self._target = makeDomain(domain)
        self._power = power

    def apply(self, x, difforder):
        self._check_input(x)
        from ..linearization import Linearization
        if difforder >= self.WITH_JAC:
            x = Linearization.make_var(x, difforder == self.WITH_METRIC)
        return x**self._power


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

    def apply(self, x, difforder):
        self._check_input(x)
        if difforder >= self.WITH_JAC:
            from ..linearization import Linearization
            x = Linearization.make_var(x, difforder == self.WITH_METRIC)
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

    def apply(self, x, difforder):
        from ..linearization import Linearization
        from ..sugar import makeOp
        self._check_input(x)
        v1 = x.extract(self._op1.domain)
        v2 = x.extract(self._op2.domain)
        if difforder == self.VALUE_ONLY:
            return self._op1(v1) * self._op2(v2)
        wm = difforder == self.WITH_METRIC
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

    def apply(self, x, difforder):
        from ..linearization import Linearization
        self._check_input(x)
        v1 = x.extract(self._op1.domain)
        v2 = x.extract(self._op2.domain)
        if difforder == self.VALUE_ONLY:
            return self._op1(v1).unite(self._op2(v2))
        wm = difforder == self.WITH_METRIC
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
