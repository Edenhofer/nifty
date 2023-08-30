# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from typing import Any, Callable, Dict, Hashable, Mapping, TypeVar, Union

import jax
from jax import numpy as jnp
from jax.tree_util import tree_map, tree_reduce

O = TypeVar('O')
I = TypeVar('I')


def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(f), primals, tangents)[1]


def split(mappable, keys):
    """Split a dictionary into one containing only the specified keys and one
    with all of the remaining ones.
    """
    sel, rest = {}, {}
    for k, v in mappable.items():
        if k in keys:
            sel[k] = v
        else:
            rest[k] = v
    return sel, rest


def isiterable(candidate):
    try:
        iter(candidate)
        return True
    except (TypeError, AttributeError):
        return False


def is1d(ls: Any) -> bool:
    """Indicates whether the input is one dimensional.

    An object is considered one dimensional if it is an iterable of
    non-iterable items.
    """
    if hasattr(ls, "ndim"):
        return ls.ndim == 1
    if not isiterable(ls):
        return False
    return all(not isiterable(e) for e in ls)


def doc_from(original):
    def wrapper(target):
        target.__doc__ = original.__doc__
        return target

    return wrapper


def ducktape(call: Callable[[I], O],
             key: Hashable) -> Callable[[Mapping[Hashable, I]], O]:
    def named_call(p):
        return call(p[key])

    return named_call


def ducktape_left(call: Callable[[I], O],
                  key: Hashable) -> Callable[[I], Dict[Hashable, O]]:
    def named_call(p):
        return {key: call(p)}

    return named_call


def sum_of_squares(tree) -> Union[jnp.ndarray, jnp.inexact]:
    return tree_reduce(jnp.add, tree_map(lambda x: jnp.sum(x**2), tree), 0.)

def sum_of_abs_squares(tree) -> Union[jnp.ndarray, jnp.inexact]:
    return tree_reduce(jnp.add, tree_map(lambda x: jnp.sum(jnp.abs(x)**2), tree), 0.)


def interpolate(xmin=-7., xmax=7., N=14000) -> Callable:
    """Replaces a local nonlinearity such as jnp.exp with a linear interpolation

    Interpolating functions speeds up code and increases numerical stability in
    some cases, but at a cost of precision and range.

    Parameters
    ----------
    xmin : float
        Minimal interpolation value. Default: -7.
    xmax : float
        Maximal interpolation value. Default: 7.
    N : int
        Number of points used for the interpolation. Default: 14000
    """
    def decorator(f):
        from functools import wraps

        x = jnp.linspace(xmin, xmax, N)
        y = f(x)

        @wraps(f)
        def wrapper(t):
            return jnp.interp(t, x, y)

        return wrapper

    return decorator
