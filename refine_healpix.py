#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from collections import namedtuple
from functools import partial
from math import log2
import sys
from typing import Callable, Optional, Tuple
import warnings

import healpy as hp
from healpy import pixelfunc
from jax import vmap
import jax
from jax import numpy as jnp
from jax import random
from jax.lax import dynamic_slice_in_dim
import matplotlib.pyplot as plt
import numpy as np

import nifty8.re as jft
from nifty8.re.refine import _get_cov_from_loc

jax.config.update("jax_debug_nans", True)


# %%
def get_1st_hp_nbrs_idx(nside, pix, nest: bool = False, dtype=np.uint32):
    from healpy import pixelfunc

    n_nbr = 8

    n_pix = 1 if np.ndim(pix) == 0 else len(pix)
    pix_nbr = np.zeros((n_pix, n_nbr + 1), dtype=int)
    pix_nbr[:, 0] = pix
    nbr = pixelfunc.get_all_neighbours(nside, pix, nest=nest)
    nbr = nbr.reshape(n_nbr, n_pix).T
    pix_nbr[:, 1:] = nbr
    pix_nbr = np.sort(pix_nbr, axis=1)  # Move `-1` to the front

    # Account for unknown neighbors, encoded by -1
    idx_w_invalid, _ = np.nonzero(nbr == -1)
    if idx_w_invalid.size != 0:
        idx_w_invalid = np.unique(idx_w_invalid)
        nbr_invalid = nbr[idx_w_invalid]
        with warnings.catch_warnings():
            wmsg = "invalid value encountered in _get_neigbors"
            warnings.filterwarnings("ignore", message=wmsg)
            # shape of (n_2nd_neighbors, n_idx_w_invalid, n_1st_neighbors)
            nbr2 = pixelfunc.get_all_neighbours(nside, nbr_invalid, nest=nest)
            nbr2 = np.transpose(nbr2, (1, 2, 0))
            nbr2[nbr_invalid == -1] = -1
            nbr2 = nbr2.reshape(idx_w_invalid.size, -1)
        pix_2nbr = np.stack(
            [
                np.setdiff1d(ar1, ar2)[:n_nbr + 1]
                for ar1, ar2 in zip(nbr2, pix_nbr[idx_w_invalid])
            ]
        )
        if np.sum(pix_2nbr == -1):
            # `setdiff1d` should remove all `-1` because we worked with rows in
            # pix_nbr that all contain them
            raise AssertionError()
        pad = max(n_nbr + 1 - pix_2nbr.shape[1], 0)
        pix_2nbr = np.pad(
            pix_2nbr, ((0, 0), (0, pad)), mode="constant", constant_values=-1
        )
        # Select a "random" 2nd neighbor to fill in for the missing 1st order
        # neighbor
        pix_nbr[idx_w_invalid] = np.where(
            pix_nbr[idx_w_invalid] == -1, pix_2nbr, pix_nbr[idx_w_invalid]
        )

    out = np.squeeze(pix_nbr, axis=0) if np.ndim(pix) == 0 else pix_nbr
    return out.astype(dtype)


def get_all_1st_hp_nbrs_idx(nside, nest: bool = False):
    pix = np.arange(12 * nside**2)
    return get_1st_hp_nbrs_idx(nside, pix, nest=nest)


def get_1st_hp_nbrs(nside, pix, nest: bool = False):
    from healpy import pixelfunc

    return np.stack(
        pixelfunc.pix2vec(
            nside, get_1st_hp_nbrs_idx(nside, pix, nest=nest), nest=nest
        ),
        axis=-1
    )


# %%
nside = 256
pix = 0


def test_uniqueness(nside, nest):
    nbr = get_all_1st_hp_nbrs_idx(nside, nest)
    n_non_uniq = np.sum(np.diff(np.sort(nbr, axis=1), axis=1) == 0, axis=1)
    np.testing.assert_equal(n_non_uniq, 0)


for nside in (1, 2):
    for n in (True, False):
        test_uniqueness(nside, n)

get_1st_hp_nbrs(nside, pix)


# %%
def _refinement_matrices(
    # level: int,
    gc_and_gf,
    # pixel_index: Optional[Iterable[int]] = None,
    kernel: Optional[Callable] = None,
    *,
    # nest: bool = False,
    coerce_fine_kernel: bool = True,
    _cov_from_loc: Optional[Callable] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    cov_from_loc = _get_cov_from_loc(kernel, _cov_from_loc)
    gc, gf = gc_and_gf
    n_fsz = gf.shape[0]

    coord = jnp.concatenate((gc, gf), axis=0)
    cov = cov_from_loc(coord, coord)
    cov_ff = cov[-n_fsz:, -n_fsz:]
    cov_fc = cov[-n_fsz:, :-n_fsz]
    cov_cc = cov[:-n_fsz, :-n_fsz]
    cov_cc_inv = jnp.linalg.inv(cov_cc)

    olf = cov_fc @ cov_cc_inv
    # Also see Schur-Complement
    fine_kernel = cov_ff - cov_fc @ cov_cc_inv @ cov_fc.T
    if coerce_fine_kernel:
        # TODO: Try to work with NaN to avoid the expensive eigendecomposition;
        # work with nan_to_num?
        # Implicitly assume a white power spectrum beyond the numerics limit.
        # Use the diagonal as estimate for the magnitude of the variance.
        fine_kernel_fallback = jnp.diag(jnp.abs(jnp.diag(fine_kernel)))
        # Never produce NaNs (https://github.com/google/jax/issues/1052)
        # This is expensive but necessary (worse but cheaper:
        # `jnp.all(jnp.diag(fine_kernel) > 0.)`)
        is_pos_def = jnp.all(jnp.linalg.eigvalsh(fine_kernel) > 0)
        fine_kernel = jnp.where(is_pos_def, fine_kernel, fine_kernel_fallback)
        # NOTE, subsequently use the Cholesky decomposition, even though
        # already having computed the eigenvalues, as to get consistent results
        # across platforms
    # Matrices are symmetrized by JAX, i.e. gradients are projected to the
    # subspace of symmetric matrices (see
    # https://github.com/google/jax/issues/10815)
    fine_kernel_sqrt = jnp.linalg.cholesky(fine_kernel)

    return olf, fine_kernel_sqrt


# %%
def matern_kernel(distance, scale, cutoff, dof):
    """Evaluates the Matern covariance kernel parametrized by its `scale`,
    length scale (a.k.a. `cutoff`) and degree-of-freedom parameter `dof` at
    `distance`.
    """
    if dof == 0.5:
        cov = scale**2 * jnp.exp(-distance / cutoff)
    elif dof == 1.5:
        reg_dist = jnp.sqrt(3) * distance / cutoff
        cov = scale**2 * (1 + reg_dist) * jnp.exp(-reg_dist)
    elif dof == 2.5:
        reg_dist = jnp.sqrt(5) * distance / cutoff
        cov = scale**2 * (1 + reg_dist + reg_dist**2 / 3) * jnp.exp(-reg_dist)
    else:
        from jax.scipy.special import gammaln
        from scipy.special import kv
        from warnings import warn

        warn("falling back to generic Matern covariance function")
        reg_dist = jnp.sqrt(2 * dof) * distance / cutoff
        cov = scale**2 * 2**(1 - dof) / jnp.exp(
            gammaln(dof)
        ) * (reg_dist)**dof * kv(dof, reg_dist)

    # NOTE, this is not safe for differentiating because `cov` still may
    # contain NaNs
    return jnp.where(distance < 1e-8 * cutoff, scale**2, cov)


def _vmap_squeeze_first_2ndax(fun, *args, **kwargs):
    vfun = vmap(fun, *args, **kwargs)

    def vfun_apply(*x):
        return vfun(jnp.squeeze(x[0], axis=1), *x[1:])

    return vfun_apply


@partial(jax.jit, static_argnames=("radial_chart", "kernel", "precision"))
def refine_slice(
    radial_chart,
    coarse_values,
    excitations,
    kernel,
    precision=None,
):
    ndim = np.ndim(coarse_values)
    if ndim not in (1, 2):
        raise ValueError(f"invalid dimensions {ndim!r}; expected either 0 or 1")
    coarse_values = coarse_values[:, np.newaxis] if ndim == 1 else coarse_values
    fsz_hp = 4
    fsz_r = 2
    csz_hp = 9
    csz_r = 3

    nside = (coarse_values.shape[0] / 12)**0.5
    level = log2(nside)
    if not nside.is_integer() or not level.is_integer():
        raise ValueError("invalid nside of `coarse_values`")
    nside, level = int(nside), int(level)

    pix_nbr_idx = get_all_1st_hp_nbrs_idx(nside, nest=nest)
    gc = np.stack(pixelfunc.pix2vec(nside, pix_nbr_idx, nest=nest), axis=-1)
    pix_idx = np.arange(coarse_values.shape[0])
    i = pixelfunc.ring2nest(nside, pix_idx) if nest is False else pix_idx
    gf = np.stack(
        pixelfunc.pix2vec(
            2 * nside, 4 * i[:, None] + np.arange(0, 4)[None, :], nest=True
        ),
        axis=-1
    )

    def refine(coarse_full, exc, idx_hp, idx_r, gc, gf):
        # `idx_r` is the left-most radial pixel of the to-be-refined slice
        # Extend `gc` and `gf` radially
        if ndim == 1:
            if gc.ndim != 2 or gf.ndim != 2:
                raise AssertionError()
        elif ndim == 2:
            bc = (1, ) * (ndim - 1) + (-1, 1)
            rc = radial_chart.ind2cart(
                idx_r + jnp.arange(csz_r)[np.newaxis, :], level
            ).reshape(bc)
            gc = gc[:, np.newaxis, :] * rc
            gc = gc.reshape(-1, ndim + 1)
            rf = radial_chart.ind2cart(
                idx_r + jnp.array([0.75, 1.25])[np.newaxis, :], level
            ).reshape(bc)
            gf = gf[:, np.newaxis, :] * rf
            gf = gf.reshape(-1, ndim + 1)
        else:
            raise AssertionError()
        olf, fks = _refinement_matrices((gc, gf), kernel=kernel)
        if ndim > 1:
            olf = olf.reshape(fsz_hp, fsz_r, csz_hp, csz_r)

        c = coarse_full[idx_hp]
        if ndim == 2:
            c = dynamic_slice_in_dim(
                coarse_full[idx_hp], idx_r, slice_size=csz_r, axis=1
            )
        refined = jnp.tensordot(olf, c, axes=ndim, precision=precision)
        f_shp = (fsz_hp, ) if ndim == 1 else (fsz_hp, fsz_r)
        refined += jnp.matmul(fks, exc, precision=precision).reshape(f_shp)
        return refined

    # TODO: benchmark swapping these two
    if ndim == 1:
        pix_r_off = None
        vrefine = _vmap_squeeze_first_2ndax(
            refine, in_axes=(None, 0, 0, None, 0, 0)
        )
    elif ndim == 2:
        pix_r_off = jnp.arange(radial_chart.shape_at(level)[0] - csz_r + 1)
        vrefine = vmap(refine, in_axes=(None, 0, None, 0, None, None))
        vrefine = vmap(vrefine, in_axes=(None, 0, 0, None, 0, 0))
    else:
        raise AssertionError()
    refined = vrefine(
        coarse_values, excitations, pix_nbr_idx, pix_r_off, gc, gf
    )
    if ndim == 1:
        refined = refined.ravel()
    elif ndim == 2:
        refined = jnp.transpose(refined, (0, 2, 1, 3))
        n_hp = refined.shape[0] * refined.shape[1]
        n_r = refined.shape[2] * refined.shape[3]
        refined = refined.reshape(n_hp, n_r)
    else:
        raise AssertionError()
    return refined


# %%
nest = True
kernel = partial(matern_kernel, scale=1., cutoff=1., dof=1.5)

pix0s = np.stack(pixelfunc.pix2vec(1, np.arange(12), nest=nest), axis=-1)
cov_from_loc = _get_cov_from_loc(kernel, None)
fks_sqrt = jnp.linalg.cholesky(cov_from_loc(pix0s, pix0s))

key = random.PRNGKey(43)
r0 = random.normal(key, (12, ))
refined = fks_sqrt @ r0
hp.mollview(refined, nest=nest)
depth = 6
key = random.PRNGKey(42)
for i in range(depth):
    _, key = random.split(key)
    exc = random.normal(key, (refined.shape[0], 4))
    refined = refine_slice(None, refined, exc, kernel)
    hp.mollview(refined, nest=nest)
# plt.show()

# %%
key = random.PRNGKey(42)


def rg2cart(x, idx0, scl):
    """Transforms regular, points from a Euclidean space to irregular points in
    an cartesian coordinate system in 1D."""
    return jnp.exp(scl * x[0] + idx0)[np.newaxis, ...]


def cart2rg(x, idx0, scl):
    """Inverse of `rg2cart`."""
    return ((jnp.log(x[0]) - idx0) / scl)[np.newaxis, ...]


n_r = 4
radial_chart = jft.CoordinateChart(
    min_shape=(n_r, ),
    depth=1,
    rg2cart=partial(rg2cart, idx0=-0.27, scl=1.1),
    cart2rg=partial(cart2rg, idx0=-0.27, scl=1.1),
    _coarse_size=3,
    _fine_size=2,
)
pix0s = np.stack(pixelfunc.pix2vec(1, np.arange(12), nest=nest), axis=-1)
pix0s = (
    pix0s[:, np.newaxis, :] *
    radial_chart.ind2cart(jnp.arange(n_r)[np.newaxis, :], -1)[..., np.newaxis]
).reshape(12 * n_r, 3)
cov_from_loc = _get_cov_from_loc(kernel, None)
fks_sqrt = jnp.linalg.cholesky(cov_from_loc(pix0s, pix0s))

r0 = random.normal(key, (12 * n_r, ))
refined = (fks_sqrt @ r0).reshape(12, n_r)
depth = 7
for i in range(depth):
    _, key = random.split(key)
    exc = random.normal(key, (refined.shape[0], refined.shape[1] - 2, 8))
    refined = refine_slice(radial_chart, refined, exc, kernel)

# %%
for i in range(refined.shape[1]):
    hp.mollview((fks_sqrt @ r0).reshape(12, n_r)[:, i], nest=nest)
    hp.mollview(refined[:, i], nest=nest)
# plt.show()

# %%
Timed = namedtuple(
    "Timed",
    ("time", "number", "repeat", "median", "min", "max", "mean", "std"),
    rename=True
)


def timeit(stmt, setup="pass", repeat=7, number=None):
    """Handy timer utility returning the median time it took evaluate `stmt`."""
    import timeit

    timer = timeit.Timer(stmt, setup=setup)
    if number is None:
        number, _ = timer.autorange()
    timings = np.array(timer.repeat(repeat=repeat, number=number)) / number

    t = np.median(timings)
    mi, ma = np.min(timings), np.max(timings)
    m, std = np.mean(timings), np.std(timings)
    return Timed(
        time=t,
        number=number,
        repeat=repeat,
        median=t,
        min=mi,
        max=ma,
        mean=m,
        std=std
    )


pix0s = np.stack(pixelfunc.pix2vec(1, np.arange(12), nest=nest), axis=-1)
cov_from_loc = _get_cov_from_loc(kernel, None)
fks_sqrt = jnp.linalg.cholesky(cov_from_loc(pix0s, pix0s))

key = random.PRNGKey(12)
r0 = random.normal(key, (12, ))
refined = fks_sqrt @ r0
depth = 9
for i in range(depth):
    _r = refined
    exc = random.normal(key, (refined.shape[0], 4))
    refined = refine_slice(None, refined, exc, kernel)
    t = timeit(lambda: refine_slice(None, _r, exc, kernel))
    print(
        f"{refined.shape=} time={t.time:4.2e} min={t.min:4.2e}",
        file=sys.stderr
    )

pix0s = np.stack(pixelfunc.pix2vec(1, np.arange(12), nest=nest), axis=-1)
pix0s = (
    pix0s[:, np.newaxis, :] *
    radial_chart.ind2cart(jnp.arange(n_r)[np.newaxis, :], -1)[..., np.newaxis]
).reshape(12 * n_r, 3)
cov_from_loc = _get_cov_from_loc(kernel, None)
fks_sqrt = jnp.linalg.cholesky(cov_from_loc(pix0s, pix0s))

r0 = random.normal(key, (12 * n_r, ))
refined = (fks_sqrt @ r0).reshape(12, n_r)
depth = 7
for i in range(depth):
    _r = refined
    _, key = random.split(key)
    exc = random.normal(key, (refined.shape[0], refined.shape[1] - 2, 8))
    refined = refine_slice(radial_chart, refined, exc, kernel)
    t = timeit(lambda: refine_slice(radial_chart, _r, exc, kernel))
    print(
        f"{refined.shape=} time={t.time:4.2e} min={t.min:4.2e}",
        file=sys.stderr
    )