#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial
import sys

import jax
from jax import random
import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import distance_matrix

import nifty8.re as jft
from nifty8.re import refine, refine_chart

pmp = pytest.mark.parametrize


def matern_kernel(distance, scale, cutoff, dof):
    from jax.scipy.special import gammaln
    from scipy.special import kv

    reg_dist = jnp.sqrt(2 * dof) * distance / cutoff
    return scale**2 * 2**(1 - dof) / jnp.exp(
        gammaln(dof)
    ) * (reg_dist)**dof * kv(dof, reg_dist)


scale, cutoff, dof = 1., 80., 3 / 2

x = jnp.logspace(-6, 11, base=jnp.e, num=int(1e+5))
y = matern_kernel(x, scale, cutoff, dof)
y = jnp.nan_to_num(y, nan=0.)
kernel = Partial(jnp.interp, xp=x, fp=y)
inv_kernel = Partial(jnp.interp, xp=y, fp=x)


@pmp("dist", (10., 20., 30., 1e+3))
def test_refinement_matrices_1d(dist, kernel=kernel):
    cov_from_loc = refine._get_cov_from_loc(kernel=kernel)

    coarse_coord = dist * jnp.array([0., 1., 2.])
    fine_coord = coarse_coord[tuple(
        jnp.array(coarse_coord.shape) // 2
    )] + (jnp.diff(coarse_coord) / jnp.array([-4., 4.]))
    cov_ff = cov_from_loc(fine_coord, fine_coord)
    cov_fc = cov_from_loc(fine_coord, coarse_coord)
    cov_cc_inv = jnp.linalg.inv(cov_from_loc(coarse_coord, coarse_coord))

    fine_kernel = cov_ff - cov_fc @ cov_cc_inv @ cov_fc.T
    fine_kernel_sqrt_diy = jnp.linalg.cholesky(fine_kernel)
    olf_diy = cov_fc @ cov_cc_inv

    olf, fine_kernel_sqrt = refine.layer_refinement_matrices(dist, kernel)

    assert_allclose(olf, olf_diy)
    assert_allclose(fine_kernel_sqrt, fine_kernel_sqrt_diy)


@pmp("seed", (12, 42, 43, 45))
@pmp("dist", (10., 20., 30., 1e+3))
def test_refinement_1d(seed, dist, kernel=kernel):
    rng = np.random.default_rng(seed)

    refs = (
        refine.refine_conv, refine.refine_conv_general, refine.refine_loop,
        refine.refine_vmap, refine.refine_loop, refine.refine_slice
    )
    cov_from_loc = refine._get_cov_from_loc(kernel=kernel)
    olf, fine_kernel_sqrt = refine.layer_refinement_matrices(dist, kernel)

    main_coord = jnp.linspace(0., 1000., 50)
    cov_sqrt = jnp.linalg.cholesky(cov_from_loc(main_coord, main_coord))
    lvl0 = cov_sqrt @ rng.normal(size=main_coord.shape)
    lvl1_exc = rng.normal(size=(2 * (lvl0.size - 2), ))

    fine_reference = refine.refine(lvl0, lvl1_exc, olf, fine_kernel_sqrt)
    eps = jnp.finfo(lvl0.dtype.type).eps
    aallclose = partial(
        assert_allclose, desired=fine_reference, rtol=6 * eps, atol=60 * eps
    )
    for ref in refs:
        print(f"testing {ref.__name__}", file=sys.stderr)
        aallclose(ref(lvl0, lvl1_exc, olf, fine_kernel_sqrt))


@pmp("seed", (12, 42))
@pmp("dist", (60., 1e+3, (80., 80.), (40., 90.), (1e+2, 1e+3, 1e+4)))
@pmp("_coarse_size", (3, 5))
@pmp("_fine_size", (2, 4))
@pmp("_fine_strategy", ("jump", "extend"))
def test_refinement_nd_cross_consistency(
    seed, dist, _coarse_size, _fine_size, _fine_strategy, kernel=kernel
):
    ndim = len(dist) if hasattr(dist, "__len__") else 1
    min_shape = (12, ) * ndim
    depth = 1
    refs = (refine.refine_conv_general, refine.refine_slice)
    kwargs = {
        "_coarse_size": _coarse_size,
        "_fine_size": _fine_size,
        "_fine_strategy": _fine_strategy
    }

    chart = refine_chart.CoordinateChart(
        min_shape, depth=depth, distances=dist, **kwargs
    )
    rfm = refine_chart.RefinementField(chart).matrices(kernel)
    xi = jft.random_like(
        random.PRNGKey(seed),
        refine_chart.RefinementField(chart).shapewithdtype
    )

    cf = partial(refine_chart.RefinementField.apply, chart=chart, kernel=rfm)
    fine_reference = cf(xi)
    eps = jnp.finfo(fine_reference.dtype.type).eps
    aallclose = partial(
        assert_allclose, desired=fine_reference, rtol=6 * eps, atol=60 * eps
    )
    for ref in refs:
        print(f"testing {ref.__name__}", file=sys.stderr)
        aallclose(cf(xi, _refine=ref))


@pmp("dist", (60., 1e+3, (80., 80.), (40., 90.), (1e+2, 1e+3, 1e+4)))
def test_refinement_fine_strategy_basic_consistency(dist, kernel=kernel):
    olf_j, ks_j = refine.layer_refinement_matrices(
        dist, kernel=kernel, _fine_size=2, _fine_strategy="jump"
    )
    olf_e, ks_e = refine.layer_refinement_matrices(
        dist, kernel=kernel, _fine_size=2, _fine_strategy="extend"
    )

    assert_allclose(olf_j, olf_e, rtol=1e-13, atol=0.)
    assert_allclose(ks_j, ks_e, rtol=1e-13, atol=0.)

    shape0 = (12, ) * len(dist) if isinstance(dist, tuple) else (12, )
    depth = 2
    olfs_j, (csq0_j, kss_j) = refine.refinement_matrices(
        shape0, depth, dist, kernel=kernel, _fine_strategy="jump"
    )
    olfs_e, (csq0_e, kss_e) = refine.refinement_matrices(
        shape0, depth, dist, kernel=kernel, _fine_strategy="extend"
    )

    assert_allclose(olfs_j, olfs_e, rtol=1e-13, atol=0.)
    assert_allclose(kss_j, kss_e, rtol=1e-13, atol=0.)
    assert_allclose(csq0_j, csq0_e, rtol=1e-13, atol=0.)


@pmp("dist", (60., 1e+3, (80., 80.), (40., 90.), (1e+2, 1e+3, 1e+4)))
@pmp("_coarse_size", (3, 5))
@pmp("_fine_size", (2, 4))
@pmp("_fine_strategy", ("jump", "extend"))
def test_refinement_covariance(
    dist, _coarse_size, _fine_size, _fine_strategy, kernel=kernel
):
    distances0 = np.atleast_1d(dist)
    ndim = len(distances0)

    cf = refine_chart.RefinementField(
        shape0=(_coarse_size, ) * ndim,
        depth=1,
        _coarse_size=_coarse_size,
        _fine_size=_fine_size,
        _fine_strategy=_fine_strategy,
        distances0=distances0,
        kernel=kernel
    )
    exc_shp = [
        jft.ShapeWithDtype((_coarse_size, ) * ndim),
        jft.ShapeWithDtype((_fine_size, ) * ndim)
    ]
    cf_shp = jax.eval_shape(cf, exc_shp)
    assert cf_shp.shape == (_fine_size, ) * ndim

    probe = jnp.zeros(cf_shp.shape)
    indices = np.indices(cf_shp.shape).reshape(ndim, -1)
    # Work around jax.linear_transpose NotImplementedError
    _, cf_T = jax.vjp(cf, jft.zeros_like(exc_shp))
    cf_cf_T = lambda x: cf(*cf_T(x))
    cov_empirical = jax.vmap(
        lambda idx: cf_cf_T(probe.at[tuple(idx)].set(1.)).ravel(),
        in_axes=1,
        out_axes=-1
    )(indices)

    pos = np.mgrid[tuple(slice(s) for s in cf_shp.shape)].astype(float)
    if _fine_strategy == "jump":
        pos *= distances0.reshape((-1, ) + (1, ) * ndim) / _fine_size
    elif _fine_strategy == "extend":
        pos *= distances0.reshape((-1, ) + (1, ) * ndim) / 2
    else:
        raise AssertionError(f"invalid `_fine_strategy`; {_fine_strategy}")
    pos = jnp.moveaxis(pos, 0, -1)
    p = pos.reshape(-1, ndim)
    dist_mat = distance_matrix(p, p)
    cov_truth = kernel(dist_mat)

    assert_allclose(cov_empirical, cov_truth, rtol=1e-14, atol=1e-15)


@pmp("seed", (12, 42, 43, 45))
@pmp("n_dim", (1, 2, 3, 4, 5))
def test_refinement_nd_shape(seed, n_dim, kernel=kernel):
    rng = np.random.default_rng(seed)

    distances = np.exp(rng.normal(size=(n_dim, )))
    cov_from_loc = refine._get_cov_from_loc(kernel=kernel)
    olf, fine_kernel_sqrt = refine.layer_refinement_matrices(distances, kernel)

    shp_i = 5
    gc = distances.reshape(n_dim, 1) * jnp.linspace(0., 1000., shp_i)
    gc = jnp.stack(jnp.meshgrid(*gc, indexing="ij"), axis=-1).reshape(-1, n_dim)
    cov_sqrt = jnp.linalg.cholesky(cov_from_loc(gc, gc))
    lvl0 = (cov_sqrt @ rng.normal(size=gc.shape[0])).reshape((shp_i, ) * n_dim)
    lvl1_exc = rng.normal(size=tuple(n - 2 for n in lvl0.shape) + (2**n_dim, ))

    fine_reference = refine.refine(lvl0, lvl1_exc, olf, fine_kernel_sqrt)
    assert fine_reference.shape == tuple((2 * (shp_i - 2), ) * n_dim)


@pmp("dist", (60., 1e+3, (80., 80.), (40., 90.), (1e+2, 1e+3, 1e+4)))
@pmp("_coarse_size", (3, 5))
@pmp("_fine_size", (2, 4))
@pmp("_fine_strategy", ("jump", "extend"))
def test_chart_pixel_refinement_matrices_consistency(
    dist, _coarse_size, _fine_size, _fine_strategy, kernel=kernel
):
    depth = 3
    distances = np.atleast_1d(dist)
    kwargs = {
        "_coarse_size": _coarse_size,
        "_fine_size": _fine_size,
        "_fine_strategy": _fine_strategy
    }

    cc = refine_chart.CoordinateChart(
        (12, ) * distances.size, depth=depth, distances=distances, **kwargs
    )
    olf, ks = refine_chart.RefinementField(cc).matrices_at(
        level=depth, pixel_index=(0, ) * distances.size, kernel=kernel
    )
    olf_classical, ks_classical = refine.layer_refinement_matrices(
        distances, kernel, **kwargs
    )
    assert_allclose(olf, olf_classical, atol=1e-14, rtol=1e-14)
    assert_allclose(ks, ks_classical, atol=1e-14, rtol=1e-14)


@pmp("dist", (60., 1e+3, (80., 80.), (40., 90.), (1e+2, 1e+3, 1e+4)))
@pmp("_coarse_size", (3, 5))
@pmp("_fine_size", (2, 4))
@pmp("_fine_strategy", ("jump", "extend"))
def test_chart_refinement_matrices_consistency(
    dist, _coarse_size, _fine_size, _fine_strategy, kernel=kernel
):
    depth = 3
    distances = np.atleast_1d(dist)
    ndim = distances.size
    kwargs = {
        "_coarse_size": _coarse_size,
        "_fine_size": _fine_size,
        "_fine_strategy": _fine_strategy
    }

    cc = refine_chart.CoordinateChart(
        (12, ) * ndim, depth=depth, distances=distances, **kwargs
    )
    refinement = refine_chart.RefinementField(cc).matrices(kernel=kernel)

    cc_irreg = refine_chart.CoordinateChart(
        shape0=cc.shape0,
        depth=depth,
        distances=distances,
        irregular_axes=tuple(range(ndim)),
        **kwargs
    )
    refinement_irreg = refine_chart.RefinementField(cc_irreg).matrices(
        kernel=kernel
    )

    _, (cov_sqrt0, _) = refine.refinement_matrices(
        cc.shape0, 0, cc.distances0, kernel, **kwargs
    )

    aallclose = partial(assert_allclose, rtol=1e-14, atol=1e-13)
    aallclose(refinement.cov_sqrt0, cov_sqrt0)
    aallclose(refinement_irreg.cov_sqrt0, cov_sqrt0)

    for lvl in range(depth):
        olf, ks = refinement.filter[lvl], refinement.propagator_sqrt[lvl]
        olf_irreg, ks_irreg = refinement_irreg.filter[
            lvl], refinement_irreg.propagator_sqrt[lvl]

        if _fine_strategy == "jump":
            distances_lvl = cc.distances0 / _fine_size**lvl
        elif _fine_strategy == "extend":
            distances_lvl = cc.distances0 / 2**lvl
        else:
            raise AssertionError()
        olf_classical, ks_classical = refine.layer_refinement_matrices(
            distances_lvl, kernel, **kwargs
        )

        aallclose(olf.squeeze(), olf_classical)
        aallclose(ks.squeeze(), ks_classical)

        olf_d = np.diff(
            olf_irreg.reshape((-1, ) + olf_irreg.shape[-2:]), axis=0
        )
        ks_d = np.diff(ks_irreg.reshape((-1, ) + ks_irreg.shape[-2:]), axis=0)
        aallclose(olf_d, 0.)
        aallclose(ks_d, 0.)
        aallclose(olf_irreg[(0, ) * ndim], olf_classical)
        aallclose(ks_irreg[(0, ) * ndim], ks_classical)


@pmp("seed", (12, ))
@pmp("dist", (60., 1e+3, (80., 80.), (40., 90.), (1e+2, 1e+3, 1e+4)))
@pmp("_coarse_size", (3, 5))
@pmp("_fine_size", (2, 4))
@pmp("_fine_strategy", ("jump", "extend"))
@pmp("_refine", (refine.refine_conv_general, refine.refine_slice))
def test_refinement_irregular_regular_consistency(
    seed,
    dist,
    _coarse_size,
    _fine_size,
    _fine_strategy,
    _refine,
    kernel=kernel
):
    depth = 1
    distances = np.atleast_1d(dist)
    ndim = distances.size
    kwargs = {
        "_coarse_size": _coarse_size,
        "_fine_size": _fine_size,
        "_fine_strategy": _fine_strategy
    }

    cc = refine_chart.RefinementField(
        shape0=(2 * _coarse_size, ) * ndim,
        depth=depth,
        distances=distances,
        **kwargs
    )
    refinement = cc.matrices(kernel=kernel)

    cc_irreg = refine_chart.RefinementField(
        shape0=cc.chart.shape0,
        depth=depth,
        distances=distances,
        irregular_axes=tuple(range(ndim)),
        **kwargs
    )
    refinement_irreg = cc_irreg.matrices(kernel=kernel)

    rng = np.random.default_rng(seed)
    exc_swd = cc.shapewithdtype[-1]
    fn1 = rng.normal(size=cc.chart.shape_at(depth - 1))
    exc = rng.normal(size=exc_swd.shape)

    refined = _refine(
        fn1, exc, refinement.filter[-1], refinement.propagator_sqrt[-1],
        **kwargs
    )
    refined_irreg = _refine(
        fn1, exc, refinement_irreg.filter[-1],
        refinement_irreg.propagator_sqrt[-1], **kwargs
    )
    assert_allclose(refined_irreg, refined, rtol=1e-14, atol=1e-13)


if __name__ == "__main__":
    test_refinement_matrices_1d(5.)
    test_refinement_1d(42, 10.)
