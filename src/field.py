from jax.numpy import add, ndim, dot
from jax.tree_util import (
    register_pytree_node_class, tree_map, tree_multimap, tree_reduce
)


@register_pytree_node_class
class Field():
    def __init__(self, val, domain=None, flags=None):
        """Value storage for arbitrary objects with added numerics.

        Parameters
        ----------
        val : object
            Arbitrary, flatten-able objects.
        domain : dict or None, optional
            Domain of the field, e.g. with description of modes and volume.
        flags : set, str or None, optional
            Capabilities or constraints of the field.
        """
        self._val = val
        self._domain = {} if domain is None else dict(domain)
        if isinstance(flags, (tuple, set)):
            self._flags = set(flags)
        elif isinstance(flags, str):
            self._flags = set(("flags", ))
        else:
            self._flags = set()

    def tree_flatten(self):
        """Recipe for flattening fields.

        Returns
        -------
        flat_tree : tuple of two tuples
            Pair of an iterable with the children to be flattened recursively,
            and some opaque auxiliary data.
        """
        return ((self._val, ), (self._domain, self._flags))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recipe to construct fields from flattened Pytrees.

        Parameters
        ----------
        aux_data : tuple of a dict and a set
            Opaque auxiliary data describing a field.
        children: tuple
            Value of the field, i.e. unflattened children.

        Returns
        -------
        unflattened_tree : Field
            Re-constructed field.
        """
        return cls(*children, domain=aux_data[0], flags=aux_data[1])

    @property
    def val(self):
        return self._val

    @property
    def domain(self):
        return self._domain.copy()

    @property
    def flags(self):
        return self._flags.copy()

    def new(self, val):
        return Field(val, domain=self.domain, flags=self.flags)

    def dot(self, other):
        if other.domain != self.domain:
            raise ValueError("domains are incompatible.")
        tree_dot = tree_multimap(
            lambda x, y: dot(x.ravel(), y.ravel()), self._val, other._val
        )
        return tree_reduce(add, tree_dot, 0.)

    def sum_of_squares(self):
        from .sugar import sum_of_squares
        return sum_of_squares(self)

    def norm(self, ord):
        from .sugar import jft_norm
        return jft_norm(self, ord=ord)

    def _val_op(self, op, *args, **kwargs):
        return getattr(self._val, op)(*args, **kwargs)

    def _unary_op(self, op):
        return self.new(tree_map(lambda c: getattr(c, op)(), self._val))

    def _binary_op(self, other, op):
        if isinstance(other, Field):
            if other.domain != self.domain:
                raise ValueError("domains are incompatible.")
        elif ndim(other) == 0:
            from jax.tree_util import tree_structure
            from itertools import repeat

            ts = tree_structure(self)
            other = ts.unflatten(repeat(other, ts.num_leaves))
        else:
            te = "Invalid binary op for Field and {}".format(type(other))
            raise TypeError(te)
        return self.new(
            tree_multimap(
                lambda s, o: getattr(s, op)(o), self._val, other._val
            )
        )


for op in ["__neg__", "__pos__", "__abs__"]:

    def func(op):
        def func2(self):
            return self._unary_op(op)

        return func2

    setattr(Field, op, func(op))

for op in [
    "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
    "__truediv__", "__rtruediv__", "__floordiv__", "__rfloordiv__", "__pow__",
    "__rpow__", "__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"
]:

    def func(op):
        def func2(self, other):
            return self._binary_op(other, op)

        return func2

    setattr(Field, op, func(op))

for op in ["get", "__getitem__", "__contains__", "__iter__", "__len__"]:

    def func(op):
        def func2(self, *args, **kwargs):
            return self._val_op(op, *args, **kwargs)

        return func2

    setattr(Field, op, func(op))
