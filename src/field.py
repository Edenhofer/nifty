from jax.tree_util import (
    register_pytree_node_class, tree_map, tree_reduce, tree_structure
)


@register_pytree_node_class
class Field():
    """Value storage for arbitrary objects with added numerics."""
    supported_flags = {"strict_domain_checking"}

    def __init__(self, val, domain=None, flags=None):
        """Instantiates a field.

        Parameters
        ----------
        val : object
            Arbitrary, flatten-able objects.
        domain : dict or None, optional
            Domain of the field, e.g. with description of modes and volume.
        flags : set, str or None, optional
            Capabilities and constraints of the field.
        """
        self._val = val
        self._domain = {} if domain is None else dict(domain)

        flags = (flags, ) if isinstance(flags, str) else flags
        flags = set() if flags is None else set(flags)
        if not flags.issubset(Field.supported_flags):
            ve = (
                f"specified flags ({flags!r}) are not a subset of the"
                f" supported flags ({Field.supported_flags!r})"
            )
            raise ValueError(ve)
        self._flags = flags

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
        """Retrieves a **view** of the field's values."""
        return self._val

    @property
    def domain(self):
        """Retrieves a **copy** of the field's domain."""
        return self._domain.copy()

    @property
    def flags(self):
        """Retrieves a **copy** of the field's flags."""
        return self._flags.copy()

    def new(self, val, domain=None, flags=None):
        """Instantiates a new field with the same domain and flags as this
        instance of a field.

        Parameters
        ----------
        val : object
            Arbitrary, flatten-able objects.
        domain : dict or None, optional
            Domain of the field, e.g. with description of modes and volume.
        flags : set, str or None, optional
            Capabilities and constraints of the field.
        """
        return Field(
            val,
            domain=self.domain if domain is None else domain,
            flags=self.flags if flags is None else flags
        )

    def dot(self, other):
        """Returns the dot product of this field with another flatten-able
        object.

        Parameters
        ----------
        other : object
            Arbitrary, flatten-able objects.

        Returns
        -------
        out : float
            Dot product of fields.
        """
        from jax.numpy import add, dot

        if isinstance(other, Field):
            if "strict_domain_checking" in self.flags | other.flags:
                if other.domain != self.domain:
                    raise ValueError("domains are incompatible.")
        tree_dot = tree_map(
            lambda x, y: dot(x.ravel(), y.ravel()), self._val, other._val
        )
        return tree_reduce(add, tree_dot, 0.)

    def __str__(self):
        s = f"Field(\n{self._val}"
        if self._domain:
            s += f",\ndomain={self._domain}"
        if self._flags:
            s += f",\nflags={self._flags}"
        s += ")"
        return s

    def __repr__(self):
        s = f"Field(\n{self._val!r}"
        if self._domain:
            s += f",\ndomain={self._domain!r}"
        if self._flags:
            s += f",\nflags={self._flags!r}"
        s += ")"
        return s

    def norm(self, ord):
        from .sugar import norm as jft_norm
        return jft_norm(self, ord=ord)

    def _val_op(self, op, *args, **kwargs):
        return getattr(self._val, op)(*args, **kwargs)

    def _unary_op(self, op):
        return self.new(tree_map(lambda c: getattr(c, op)(), self._val))

    def _binary_op(self, other, op):
        from jax.numpy import ndim

        flags = None
        if isinstance(other, Field):
            flags = self.flags | other.flags
            if "strict_domain_checking" in flags:
                if other.domain != self.domain:
                    raise ValueError("domains are incompatible.")
        elif ndim(other) == 0:
            from itertools import repeat

            ts = tree_structure(self)
            other = ts.unflatten(repeat(other, ts.num_leaves))
        else:
            te = "Invalid binary op for Field and {}".format(type(other))
            raise TypeError(te)
        return self.new(
            tree_map(lambda s, o: getattr(s, op)(o), self._val, other._val),
            flags=flags
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
