#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from math import ceil
import sys
from typing import Callable, Iterable, Literal, Optional, Tuple, Union
from warnings import warn

import jax
from jax import numpy as jnp
import numpy as np
from scipy.spatial import distance_matrix

from .forest_util import zeros_like


def coarse2fine_shape(
    shape0: Union[int, Iterable[int]],
    depth: int,
    *,
    _coarse_size: int = 3,
    _fine_size: int = 2,
    _fine_strategy: Literal["jump", "extend"] = "jump",
):
    """Translates a coarse shape to its corresponding fine shape."""
    shape0 = (shape0, ) if isinstance(shape0, int) else shape0
    csz = int(_coarse_size)  # coarse size
    fsz = int(_fine_size)  # fine size
    if _fine_size % 2 != 0:
        raise ValueError("only even numbers allowed for `_fine_size`")

    shape = []
    for shp in shape0:
        sz_at = shp
        for lvl in range(depth):
            if _fine_strategy == "jump":
                sz_at = fsz * (sz_at - (csz - 1))
            elif _fine_strategy == "extend":
                sz_at = fsz * ceil((sz_at - (csz - 1)) / (fsz // 2))
            else:
                ve = f"invalid `_fine_strategy`; got {_fine_strategy}"
                raise ValueError(ve)
            if sz_at <= 0:
                ve = (
                    f"`shape0` ({shape0}) with `depth` ({depth}) yield an"
                    f" invalid shape ({sz_at}) at level {lvl}"
                )
                raise ValueError(ve)
        shape.append(int(sz_at))
    return tuple(shape)


def fine2coarse_shape(
    shape: Union[int, Iterable[int]],
    depth: int,
    *,
    _coarse_size: int = 3,
    _fine_size: int = 2,
    _fine_strategy: Literal["jump", "extend"] = "jump",
    ceil_sizes: bool = False,
):
    """Translates a fine shape to its corresponding coarse shape."""
    shape = (shape, ) if isinstance(shape, int) else shape
    csz = int(_coarse_size)  # coarse size
    fsz = int(_fine_size)  # fine size
    if _fine_size % 2 != 0:
        raise ValueError("only even numbers allowed for `_fine_size`")

    shape0 = []
    for shp in shape:
        sz_at = shp
        for lvl in range(depth, 0, -1):
            if _fine_strategy == "jump":
                # solve for n: `fsz * (n - (csz - 1))`
                sz_at = sz_at / fsz + (csz - 1)
            elif _fine_strategy == "extend":
                # solve for n: `fsz * ceil((n - (csz - 1)) / (fsz // 2))`
                # NOTE, not unique because of `ceil`; use lower limit
                sz_at_max = (sz_at / fsz) * (fsz // 2) + (csz - 1)
                sz_at_min = ceil(sz_at_max - (fsz // 2 - 1))
                for sz_at_cand in range(sz_at_min, ceil(sz_at_max) + 1):
                    try:
                        shp_cand = coarse2fine_shape(
                            (sz_at_cand, ),
                            depth=depth - lvl + 1,
                            _coarse_size=csz,
                            _fine_size=fsz,
                            _fine_strategy=_fine_strategy
                        )[0]
                    except ValueError as e:
                        if "invalid shape" not in "".join(e.args):
                            ve = "unexpected behavior of `coarse2fine_shape`"
                            raise ValueError(ve) from e
                        shp_cand = -1
                    if shp_cand >= shp:
                        sz_at = sz_at_cand
                        break
                else:
                    ve = f"interval search within [{sz_at_min}, {ceil(sz_at_max)}] failed"
                    raise ValueError(ve)
            else:
                ve = f"invalid `_fine_strategy`; got {_fine_strategy}"
                raise ValueError(ve)

            sz_at = ceil(sz_at) if ceil_sizes else sz_at
            if sz_at != int(sz_at):
                raise ValueError(f"invalid shape at level {lvl}")
        shape0.append(int(sz_at))
    return tuple(shape0)


def coarse2fine_distances(
    distances0: Union[float, Iterable[float]],
    depth: int,
    *,
    _fine_size: int = 2,
    _fine_strategy: Literal["jump", "extend"] = "jump",
):
    """Translates coarse distances to its corresponding fine distances."""
    fsz = int(_fine_size)  # fine size
    if _fine_strategy == "jump":
        fpx_in_cpx = fsz**depth
    elif _fine_strategy == "extend":
        fpx_in_cpx = 2**depth
    else:
        ve = f"invalid `_fine_strategy`; got {_fine_strategy}"
        raise ValueError(ve)

    return jnp.atleast_1d(distances0) / fpx_in_cpx


def fine2coarse_distances(
    distances: Union[float, Iterable[float]],
    depth: int,
    *,
    _fine_size: int = 2,
    _fine_strategy: Literal["jump", "extend"] = "jump",
):
    """Translates fine distances to its corresponding coarse distances."""
    fsz = int(_fine_size)  # fine size
    if _fine_strategy == "jump":
        fpx_in_cpx = fsz**depth
    elif _fine_strategy == "extend":
        fpx_in_cpx = 2**depth
    else:
        ve = f"invalid `_fine_strategy`; got {_fine_strategy}"
        raise ValueError(ve)

    return jnp.atleast_1d(distances) * fpx_in_cpx


def gauss_kl(cov_desired, cov_approx, *, m_desired=None, m_approx=None):
    cov_t_ds, cov_t_dl = jnp.linalg.slogdet(cov_desired)
    cov_a_ds, cov_a_dl = jnp.linalg.slogdet(cov_approx)
    if (cov_t_ds * cov_a_ds) <= 0.:
        raise ValueError("fraction of determinants must be positive")

    cov_a_inv = jnp.linalg.inv(cov_approx)

    kl = -cov_desired.shape[0]  # number of dimensions
    kl += cov_a_dl - cov_t_dl + jnp.trace(cov_a_inv @ cov_desired)
    if m_approx is not None and m_desired is not None:
        m_diff = m_approx - m_desired
        kl += m_diff @ cov_a_inv @ m_diff
    elif not (m_approx is None and m_approx is None):
        ve = "either both or neither of `m_approx` and `m_desired` must be `None`"
        raise ValueError(ve)
    return 0.5 * kl


def refinement_covariance(chart, kernel, jit=True):
    """Computes the implied covariance as modeled by the refinement scheme."""
    from .refine_chart import RefinementField

    cf = RefinementField(chart, kernel=kernel)
    try:
        cf_T = jax.linear_transpose(cf, cf.shapewithdtype)
        cov_implicit = lambda x: cf(*cf_T(x))
        cov_implicit = jax.jit(cov_implicit) if jit else cov_implicit
        _ = cov_implicit(jnp.zeros(chart.shape))  # Test transpose
    except NotImplementedError:
        # Workaround JAX not yet implementing the transpose of the scanned
        # refinement
        _, cf_T = jax.vjp(cf, zeros_like(cf.shapewithdtype))
        cov_implicit = lambda x: cf(*cf_T(x))
        cov_implicit = jax.jit(cov_implicit) if jit else cov_implicit

    probe = jnp.zeros(chart.shape)
    indices = np.indices(chart.shape).reshape(chart.ndim, -1)
    cov_empirical = jax.lax.map(
        lambda idx: cov_implicit(probe.at[tuple(idx)].set(1.)).ravel(),
        indices.T
    ).T  # vmap over `indices` w/ `in_axes=1, out_axes=-1`

    return cov_empirical


def true_covariance(chart, kernel, depth=None):
    """Computes the true covariance at the final grid."""
    depth = chart.depth if depth is None else depth

    c0 = [jnp.arange(sz) for sz in chart.shape_at(depth)]
    pos = jnp.stack(
        chart.ind2cart(jnp.meshgrid(*c0, indexing="ij"), depth), axis=0
    )

    p = jnp.moveaxis(pos, 0, -1).reshape(-1, chart.ndim)
    dist_mat = distance_matrix(p, p)
    return kernel(dist_mat)


def refinement_approximation_error(
    chart,
    kernel: Callable,
    cutout: Optional[Union[slice, int, Tuple[slice], Tuple[int]]] = None,
):
    """Computes the Kullback-Leibler (KL) divergence of the true covariance versus the
    approximative one for a given kernel and shape of the fine grid.

    If the desired shape can not be matched, the next larger one is used and
    the field is subsequently cropped to the desired shape.
    """

    suggested_min_shape = 2 * 4**chart.depth
    if any(s <= suggested_min_shape for s in chart.shape):
        msg = (
            f"shape {chart.shape} potentially too small"
            f" (desired {(suggested_min_shape, ) * chart.ndim} (=`2*4^depth`))"
        )
        warn(msg)

    cov_empirical = refinement_covariance(chart, kernel)
    cov_truth = true_covariance(chart, kernel)

    if cutout is None and all(s > suggested_min_shape for s in chart.shape):
        cutout = (suggested_min_shape, ) * chart.ndim
        print(
            f"cropping field (w/ shape {chart.shape}) to {cutout}",
            file=sys.stderr
        )
    if cutout is not None:
        if isinstance(cutout, slice):
            cutout = (cutout, ) * chart.ndim
        elif isinstance(cutout, int):
            cutout = (slice(cutout), ) * chart.ndim
        elif isinstance(cutout, tuple):
            if all(isinstance(el, slice) for el in cutout):
                pass
            elif all(isinstance(el, int) for el in cutout):
                cutout = tuple(slice(el) for el in cutout)
            else:
                raise TypeError("elements of `cutout` of invalid type")
        else:
            raise TypeError("`cutout` of invalid type")

        cov_empirical = cov_empirical.reshape(chart.shape * 2)[cutout * 2]
        cov_truth = cov_truth.reshape(chart.shape * 2)[cutout * 2]
        sz = np.prod(cov_empirical.shape[:chart.ndim])
        if np.prod(cov_truth.shape[:chart.ndim]) != sz or not sz.dtype == int:
            raise AssertionError()
        cov_empirical = cov_empirical.reshape(sz, sz)
        cov_truth = cov_truth.reshape(sz, sz)

    aux = {
        "cov_empirical": cov_empirical,
        "cov_truth": cov_truth,
    }
    return gauss_kl(cov_truth, cov_empirical), aux
