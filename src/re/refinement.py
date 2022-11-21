#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from collections import namedtuple
from functools import partial
from typing import Callable, Iterable, Optional, Tuple, Union

from jax import numpy as jnp
from jax import vmap

from .forest_util import ShapeWithDtype
from .model import AbstractModel
from .refine import _get_cov_from_loc, refine
from .refine_chart import CoordinateChart, HEALPixChart
from .refine_healpix import refine as refine_hp
from .refine_healpix import cov_sqrt as cov_sqrt_hp
from .refine_util import get_refinement_shapewithdtype

RefinementMatrices = namedtuple(
    "RefinementMatrices", ("filter", "propagator_sqrt", "cov_sqrt0")
)


def _coordinate_pixel_refinement_matrices(
    chart: CoordinateChart,
    level: int,
    pixel_index: Optional[Iterable[int]] = None,
    kernel: Optional[Callable] = None,
    *,
    coerce_fine_kernel: bool = True,
    _cov_from_loc: Optional[Callable] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    cov_from_loc = _get_cov_from_loc(kernel, _cov_from_loc)
    csz = int(chart.coarse_size)  # coarse size
    if csz % 2 != 1:
        raise ValueError("only odd numbers allowed for `_coarse_size`")
    fsz = int(chart.fine_size)  # fine size
    if fsz % 2 != 0:
        raise ValueError("only even numbers allowed for `_fine_size`")
    ndim = chart.ndim
    if pixel_index is None:
        pixel_index = (0, ) * ndim
    pixel_index = jnp.asarray(pixel_index)
    if pixel_index.size != ndim:
        ve = f"`pixel_index` has {pixel_index.size} dimensions but `chart` has {ndim}"
        raise ValueError(ve)

    csz_half = int((csz - 1) / 2)
    gc = jnp.arange(-csz_half, csz_half + 1, dtype=float)
    gc = jnp.ones((ndim, 1)) * gc
    gc = jnp.stack(jnp.meshgrid(*gc, indexing="ij"), axis=-1)
    if chart.fine_strategy == "jump":
        gf = jnp.arange(fsz, dtype=float) / fsz - 0.5 + 0.5 / fsz
    elif chart.fine_strategy == "extend":
        gf = jnp.arange(fsz, dtype=float) / 2 - 0.25 * (fsz - 1)
    else:
        raise ValueError(f"invalid `_fine_strategy`; got {chart.fine_strategy}")
    gf = jnp.ones((ndim, 1)) * gf
    gf = jnp.stack(jnp.meshgrid(*gf, indexing="ij"), axis=-1)
    # On the GPU a single `cov_from_loc` call is about twice as fast as three
    # separate calls for coarse-coarse, fine-fine and coarse-fine.
    coord = jnp.concatenate(
        (gc.reshape(-1, ndim), gf.reshape(-1, ndim)), axis=0
    )
    coord = chart.ind2cart((coord + pixel_index.reshape((1, ndim))).T, level)
    coord = jnp.stack(coord, axis=-1)
    del gc, gf
    cov = cov_from_loc(coord, coord)
    del coord
    cov_ff = cov[-fsz**ndim:, -fsz**ndim:]
    cov_fc = cov[-fsz**ndim:, :-fsz**ndim]
    cov_cc = cov[:-fsz**ndim, :-fsz**ndim]
    del cov
    cov_cc_inv = jnp.linalg.inv(cov_cc)
    del cov_cc

    olf = cov_fc @ cov_cc_inv
    # Also see Schur-Complement
    fine_kernel = cov_ff - cov_fc @ cov_cc_inv @ cov_fc.T
    del cov_cc_inv, cov_fc, cov_ff
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


def _coordinate_refinement_matrices(
    chart: CoordinateChart,
    kernel: Callable,
    *,
    depth: Optional[int] = None,
    skip0=False,
    coerce_fine_kernel: bool = True,
    _cov_from_loc=None
) -> RefinementMatrices:
    cov_from_loc = _get_cov_from_loc(kernel, _cov_from_loc)
    depth = chart.depth if depth is None else depth

    if not skip0:
        rg0 = jnp.mgrid[tuple(slice(s) for s in chart.shape0)]
        c0 = jnp.stack(chart.ind2cart(rg0, 0), axis=-1).reshape(-1, chart.ndim)
        # Matrices are symmetrized by JAX, i.e. gradients are projected to the
        # subspace of symmetric matrices (see
        # https://github.com/google/jax/issues/10815)
        cov_sqrt0 = jnp.linalg.cholesky(cov_from_loc(c0, c0))
    else:
        cov_sqrt0 = None

    opt_lin_filter, kernel_sqrt = [], []
    olf_at = vmap(
        partial(
            _coordinate_pixel_refinement_matrices,
            chart,
            coerce_fine_kernel=coerce_fine_kernel,
            _cov_from_loc=cov_from_loc,
        ),
        in_axes=(None, 0),
        out_axes=(0, 0)
    )

    for lvl in range(depth):
        shape_lvl = chart.shape_at(lvl)
        pixel_indices = []
        for ax in range(chart.ndim):
            pad = (chart.coarse_size - 1) / 2
            if int(pad) != pad:
                raise ValueError("`coarse_size` must be odd")
            pad = int(pad)
            if chart.fine_strategy == "jump":
                stride = 1
            elif chart.fine_strategy == "extend":
                stride = chart.fine_size / 2
                if int(stride) != stride:
                    raise ValueError("`fine_size` must be even")
                stride = int(stride)
            else:
                raise AssertionError()
            if ax in chart.irregular_axes:
                pixel_indices.append(
                    jnp.arange(pad, shape_lvl[ax] - pad, stride)
                )
            else:
                pixel_indices.append(jnp.array([pad]))
        pixel_indices = jnp.stack(
            jnp.meshgrid(*pixel_indices, indexing="ij"), axis=-1
        )
        shape_filtered_lvl = pixel_indices.shape[:-1]
        pixel_indices = pixel_indices.reshape(-1, chart.ndim)

        olf, ks = olf_at(lvl, pixel_indices)
        shape_bc_lvl = tuple(
            shape_filtered_lvl[i] if i in chart.irregular_axes else 1
            for i in range(chart.ndim)
        )
        opt_lin_filter.append(olf.reshape(shape_bc_lvl + olf.shape[-2:]))
        kernel_sqrt.append(ks.reshape(shape_bc_lvl + ks.shape[-2:]))

    return RefinementMatrices(opt_lin_filter, kernel_sqrt, cov_sqrt0)


class RefinementField(AbstractModel):
    def __init__(
        self,
        *args,
        kernel: Optional[Callable] = None,
        dtype=None,
        skip0: bool = False,
        **kwargs
    ):
        """Initialize an Iterative Charted Refinement (ICR) field.

        There are multiple ways to initialize a charted refinement field. The
        recommended way is to first instantiate a `CoordinateChart` and pass it
        as first argument to this method. Alternatively, you may pass any and
        all arguments of `CoordinateChart` also to this method and it will
        instantiate the `CoordinateChart` for you and use it in the same way as
        if directly specified.

        Parameters
        ----------
        chart : CoordinateChart
            The `CoordinateChart` with which to refine.
        kernel :
            Covariance kernel of the refinement field.
        dtype :
            Data-type of the excitations which to add during refining.
        skip0 :
            Whether to skip the first refinement level. This is useful to e.g.
            stack multiple refinement fields on top of each other.
        **kwargs :
            Alternatively to `chart` any parameters accepted by
            `CoordinateChart`.
        """
        self._kernel = kernel
        self._dtype = dtype
        self._skip0 = skip0

        if len(args) > 0 and isinstance(args[0], CoordinateChart):
            if kwargs:
                raise TypeError(f"expected no keyword arguments, got {kwargs}")

            if len(args) == 1:
                self._chart, = args
            elif len(args) == 2 and callable(args[1]) and kernel is None:
                self._chart, self._kernel = args
            elif len(args) == 3 and callable(
                args[1]
            ) and kernel is None and dtype is None:
                self._chart, self._kernel, self._dtype = args
            elif len(args) == 4 and callable(
                args[1]
            ) and kernel is None and dtype is None and skip0 == False:
                self._chart, self._kernel, self._dtype, self._skip0 = args
            else:
                te = "got unexpected arguments in addition to CoordinateChart"
                raise TypeError(te)
        else:
            self._chart = CoordinateChart(*args, **kwargs)

    @property
    def kernel(self):
        """Yields the kernel specified during initialization or throw a
        `TypeError`.
        """
        if self._kernel is None:
            te = (
                "either specify a fixed kernel during initialization of the"
                f" {self.__class__.__name__} class or provide one here"
            )
            raise TypeError(te)
        return self._kernel

    @property
    def dtype(self):
        """Yields the data-type of the excitations."""
        return jnp.float64 if self._dtype is None else self._dtype

    @property
    def skip0(self):
        """Whether to skip the zeroth refinement"""
        return self._skip0

    @property
    def chart(self):
        """Associated `CoordinateChart` with which to iterative refine."""
        return self._chart

    def matrices(
        self,
        kernel: Optional[Callable] = None,
        depth: Optional[int] = None,
        skip0: Optional[bool] = None,
        **kwargs
    ) -> RefinementMatrices:
        """Computes the refinement matrices namely the optimal linear filter
        and the square root of the information propagator (a.k.a. the square
        root of the fine covariance matrix for the excitations) for all
        refinement levels and all pixel indices in the coordinate chart.

        Parameters
        ----------
        kernel :
            Covariance kernel of the refinement field if not specified during
            initialization.
        depth :
            Maximum refinement depth if different to the one of the `CoordinateChart`.
        skip0 :
            Whether to skip the first refinement level.
        """
        if kernel is None and "_cov_from_loc" not in kwargs:
            kernel = self.kernel
        depth = self.chart.depth if depth is None else depth
        skip0 = self.skip0 if skip0 is None else skip0

        return _coordinate_refinement_matrices(
            self.chart, kernel=kernel, depth=depth, skip0=skip0, **kwargs
        )

    def matrices_at(
        self,
        level: int,
        pixel_index: Optional[Iterable[int]] = None,
        kernel: Optional[Callable] = None,
        **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Computes the refinement matrices namely the optimal linear filter
        and the square root of the information propagator (a.k.a. the square
        root of the fine covariance matrix for the excitations) at the
        specified level and pixel index.

        Parameters
        ----------
        level :
            Refinement level.
        pixel_index :
            Index of the NDArray at the refinement level `level` which to
            refine, i.e. use as center coarse pixel.
        kernel :
            Covariance kernel of the refinement field if not specified during
            initialization.
        """
        if kernel is None and "_cov_from_loc" not in kwargs:
            kernel = self.kernel

        return _coordinate_pixel_refinement_matrices(
            self.chart,
            level=level,
            pixel_index=pixel_index,
            kernel=kernel,
            **kwargs
        )

    @property
    def domain(self):
        """Yields the `ShapeWithDtype` of the primals."""
        return get_refinement_shapewithdtype(
            shape0=self.chart.shape0,
            depth=self.chart.depth,
            dtype=self.dtype,
            skip0=self.skip0,
            _coarse_size=self.chart.coarse_size,
            _fine_size=self.chart.fine_size,
            _fine_strategy=self.chart.fine_strategy,
        )

    @staticmethod
    def apply(
        xi,
        chart,
        kernel: Union[Callable, RefinementMatrices],
        *,
        skip0: bool = False,
        depth: Optional[int] = None,
        coerce_fine_kernel: bool = True,
        _refine: Optional[Callable] = None,
        _cov_from_loc: Optional[Callable] = None,
        precision=None,
    ):
        """Static method to apply a refinement field given some excitations, a
        chart and a kernel.

        Parameters
        ----------
        xi :
            Latent parameters which to use for refining.
        chart :
            Chart with which to refine.
        kernel :
            Covariance kernel with which to build the refinement matrices.
        skip0 :
            Whether to skip the first refinement level.
        depth :
            Refinement depth if different to the depth of the coordinate chart.
        coerce_fine_kernel :
            Whether to coerce the refinement matrices at scales at which the
            kernel matrix becomes singular or numerically highly unstable.
        precision :
            See JAX's precision.
        """
        depth = chart.depth if depth is None else depth
        if depth != len(xi) - 1:
            ve = (
                f"incompatible refinement depths of `xi` ({len(xi) - 1})"
                f" and `depth` (of chart) {depth}"
            )
            raise ValueError(ve)

        if isinstance(kernel, RefinementMatrices):
            refinement = kernel
        else:
            refinement = _coordinate_refinement_matrices(
                chart,
                kernel=kernel,
                depth=depth,
                skip0=skip0,
                coerce_fine_kernel=coerce_fine_kernel,
                _cov_from_loc=_cov_from_loc,
            )
        refine_w_chart = partial(
            refine if _refine is None else _refine,
            _coarse_size=chart.coarse_size,
            _fine_size=chart.fine_size,
            _fine_strategy=chart.fine_strategy,
            precision=precision
        )

        if not skip0:
            fine = (refinement.cov_sqrt0 @ xi[0].ravel()).reshape(xi[0].shape)
        else:
            if refinement.cov_sqrt0 is not None:
                raise AssertionError()
            fine = xi[0]
        for x, olf, k in zip(
            xi[1:], refinement.filter, refinement.propagator_sqrt
        ):
            fine = refine_w_chart(fine, x, olf, k)
        return fine

    def __call__(self, xi, kernel=None, *, skip0=None, **kwargs):
        """See `RefinementField.apply`."""
        if kernel is None and "_cov_from_loc" not in kwargs:
            kernel = self.kernel
        skip0 = self.skip0 if skip0 is None else skip0
        return self.apply(xi, self.chart, kernel=kernel, skip0=skip0, **kwargs)

    def __repr__(self):
        descr = f"{self.__class__.__name__}({self.chart!r}"
        descr += f", kernel={self._kernel!r}" if self._kernel is not None else ""
        descr += f", dtype={self._dtype!r}" if self._dtype is not None else ""
        descr += f", skip0={self.skip0!r}" if self.skip0 is not False else ""
        descr += ")"
        return descr

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))


def _healpix_pixel_refinement_matrices(
    gc_and_gf,
    kernel: Optional[Callable] = None,
    *,
    coerce_fine_kernel: bool = True,
    _cov_from_loc: Optional[Callable] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    cov_from_loc = _get_cov_from_loc(kernel, _cov_from_loc)
    n_fsz = gc_and_gf[1].shape[0]

    coord = jnp.concatenate(gc_and_gf, axis=0)
    del gc_and_gf
    cov = cov_from_loc(coord, coord)
    del coord
    cov_ff = cov[-n_fsz:, -n_fsz:]
    cov_fc = cov[-n_fsz:, :-n_fsz]
    cov_cc = cov[:-n_fsz, :-n_fsz]
    del cov
    cov_cc_inv = jnp.linalg.inv(cov_cc)
    del cov_cc

    olf = cov_fc @ cov_cc_inv
    # Also see Schur-Complement
    fine_kernel = cov_ff - cov_fc @ cov_cc_inv @ cov_fc.T
    del cov_cc_inv, cov_fc, cov_ff
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


class RefinementHPField(AbstractModel):
    def __init__(
        self,
        chart: HEALPixChart,
        kernel: Optional[Callable] = None,
        dtype=None,
    ):
        """Initialize an Iterative Charted Refinement (ICR) field for a HEALPix
        map with a radial extent.

        Parameters
        ----------
        chart :
            HEALPix coordinate chart with which to iteratively refine.
        kernel :
            Covariance kernel of the refinement field.
        dtype :
            Data-type of the excitations which to add during refining.
        """
        self._chart = chart
        self._kernel = kernel
        self._dtype = dtype

    @property
    def kernel(self):
        """Yields the kernel specified during initialization or throw a
        `TypeError`.
        """
        if self._kernel is None:
            te = (
                "either specify a fixed kernel during initialization of the"
                f" {self.__class__.__name__} class or provide one here"
            )
            raise TypeError(te)
        return self._kernel

    @property
    def dtype(self):
        """Yields the data-type of the excitations."""
        return jnp.float64 if self._dtype is None else self._dtype

    @property
    def chart(self):
        """Associated `HEALPixChart` with which to iteratively refine."""
        return self._chart

    def matrices(
        self,
        kernel: Optional[Callable] = None,
        depth: Optional[int] = None,
        coerce_fine_kernel: bool = True,
    ) -> RefinementMatrices:
        """Computes the refinement matrices namely the optimal linear filter
        and the square root of the information propagator (a.k.a. the square
        root of the fine covariance matrix for the excitations) for all
        refinement levels and all pixel indices in the coordinate chart.

        Parameters
        ----------
        kernel :
            Covariance kernel of the refinement field if not specified during
            initialization.
        depth :
            Maximum refinement depth if different to the one of the
            `HEALPixChart`.
        skip0 :
            Whether to skip the first refinement level.
        """
        cc = self.chart
        kernel = self.kernel if kernel is None else kernel
        depth = cc.depth if depth is None else depth

        def mat(lvl, idx_hp, idx_r):
            # `idx_r` is the left-most radial pixel of the to-be-refined slice
            # Extend `gc` and `gf` radially
            gc, gf = cc.get_coarse_fine_pair((idx_hp, idx_r), lvl)
            olf, ks = _healpix_pixel_refinement_matrices(
                (gc, gf), kernel=kernel, coerce_fine_kernel=coerce_fine_kernel
            )
            if cc.ndim > 1:
                olf = olf.reshape(
                    cc.fine_size**2, cc.fine_size, cc.coarse_size**2,
                    cc.coarse_size
                )
            return olf, ks

        cov_sqrt0 = cov_sqrt_hp(cc, kernel)

        opt_lin_filter, kernel_sqrt = [], []
        for lvl in range(depth):
            pix_hp_idx = jnp.arange(cc.shape_at(lvl)[0])
            if cc.ndim == 1:
                pix_r_off = None
                vmat = vmap(
                    partial(mat, lvl), in_axes=(0, 0)
                )
            elif cc.ndim == 2:
                pix_r_off = jnp.arange(cc.shape_at(lvl)[1] - cc.coarse_size + 1)
                vmat = vmap(partial(mat, lvl), in_axes=(None, 0))
                vmat = vmap(vmat, in_axes=(0, None))
            else:
                raise AssertionError()
            olf, ks = vmat(pix_hp_idx, pix_r_off)
            opt_lin_filter.append(olf)
            kernel_sqrt.append(ks)

        return RefinementMatrices(opt_lin_filter, kernel_sqrt, cov_sqrt0)

    @property
    def domain(self):
        """Yields the `ShapeWithDtype` of the primals."""
        nonhp_domain = get_refinement_shapewithdtype(
            shape0=self.chart.shape0[1:],
            depth=self.chart.depth,
            dtype=self.dtype,
            skip0=False,
            _coarse_size=self.chart.coarse_size,
            _fine_size=self.chart.fine_size,
            _fine_strategy=self.chart.fine_strategy,
        )
        domain = [
            ShapeWithDtype(
                (12 * self.chart.nside0**2, ) + nonhp_domain[0].shape,
                nonhp_domain[0].dtype
            )
        ]
        domain += [
            ShapeWithDtype(
                (12 * self.chart.nside_at(lvl)**2, ) + swd.shape[:-1] +
                (4 * swd.shape[-1], ), swd.dtype
            )
            for lvl, swd in zip(range(self.chart.depth + 1), nonhp_domain[1:])
        ]
        return domain

    @staticmethod
    def apply(
        xi,
        chart: HEALPixChart,
        kernel: Union[Callable, RefinementMatrices],
        *,
        coerce_fine_kernel: bool = True,
        _refine: Optional[Callable] = None,
        precision=None,
    ):
        """Static method to apply a refinement field given some excitations, a
        chart and a kernel.

        Parameters
        ----------
        xi :
            Latent parameters which to use for refining.
        radial_chart :
            Chart with which to refine the radial axis of the HEALPix map.
        kernel :
            Covariance kernel with which to build the refinement matrices.
        coerce_fine_kernel :
            Whether to coerce the refinement matrices at scales at which the
            kernel matrix becomes singular or numerically highly unstable.
        precision :
            See JAX's precision.
        """
        if xi[0].shape != chart.shape0:
            ve = "zeroth excitations do not fit to chart"
            raise ValueError(ve)

        if isinstance(kernel, RefinementMatrices):
            refinement = kernel
        else:
            refinement = RefinementHPField(chart, None, xi[0].dtype).matrices(
                kernel=kernel,
                coerce_fine_kernel=coerce_fine_kernel
            )
        refine_w_chart = partial(
            refine_hp if _refine is None else _refine,
            chart=chart,
            precision=precision
        )

        fine = (refinement.cov_sqrt0 @ xi[0].ravel()).reshape(xi[0].shape)
        for x, olf, k in zip(xi[1:], refinement.filter, refinement.propagator_sqrt):
            fine = refine_w_chart(fine, x, olf, k)
        return fine

    def __call__(self, xi, kernel=None, **kwargs):
        """See `RefinementField.apply`."""
        kernel = self.kernel if kernel is None else kernel
        return self.apply(xi, self.chart, kernel=kernel, **kwargs)

    def __repr__(self):
        descr = f"{self.__class__.__name__}(chart={self.chart!r}"
        descr += f", kernel={self._kernel!r}" if self._kernel is not None else ""
        descr += f", dtype={self._dtype!r}" if self._dtype is not None else ""
        descr += ")"
        return descr

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))
