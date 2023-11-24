#!/usr/bin/env python3

# SPDX-License-Identifier: BSD-2-Clause
# Authors: Philipp Frank, Jakob Roth, Gordian Edenhofer

import os
import pickle
from functools import partial, reduce
from os import makedirs
from os.path import isfile
from typing import (
    Any, Callable, List, Literal, NamedTuple, Optional, Tuple, TypeVar, Union
)

import jax
from jax import Array
from jax import numpy as jnp
from jax import random, tree_map
from jax.tree_util import Partial

from . import conjugate_gradient
from .evi import Samples, _parse_jit, curve_residual, draw_linear_residual
from .likelihood import Likelihood, StandardHamiltonian
from .logger import logger
from .misc import minisanity
from .optimize import OptimizeResults, minimize
from .tree_math import get_map
from .tree_math.vector import Vector

P = TypeVar("P")


def _optimize_kl_status_print(iiter, samples, state, residual, out_dir=None):
    en = state.minimization_state.fun
    msg = f"Post VI Iteration {iiter}: Energy {en:2.4e}\n"
    if state.sampling_states is not None:
        niter = tuple(ss.nit for ss in state.sampling_states)
        msg += f"Nonlinear sampling total iterations: {niter}\n"
    _, minis_r = minisanity(samples.pos, samples, residual)
    _, mini_pr = minisanity(samples.pos, samples)
    msg += (
        f"KL-Minimization total iteration: {state.minimization_state.nit}"
        f"\nLikelihood residual(s):\n{minis_r}"
        f"\nPrior residual(s):\n{mini_pr}"
        f"\n"
    )
    logger.info(msg)

    if not out_dir == None:
        lfile = os.path.join(out_dir, "minisanity")
        if isfile(lfile) and iiter != 0:
            with open(lfile) as f:
                msg = str(f.read()) + "\n" + msg
        with open(os.path.join(out_dir, "minisanity"), "w") as f:
            f.write(msg)


def _make_callable(obj):
    if callable(obj) and not isinstance(obj, Likelihood):
        return obj
    else:
        return lambda x: obj


def _getitem(cfg, i):
    if not isinstance(cfg, dict):
        return cfg(i)
    return {kk: _getitem(ii, i) for kk, ii in cfg.items()}


def _do_resample(cfg, iiter):
    if iiter == 0:
        return True
    cfgi = _getitem(cfg, iiter)
    cfgo = _getitem(cfg, iiter - 1)
    regenerate = cfgi['resample']
    regenerate += (cfgi['n_samples'] != cfgo['n_samples'])
    return bool(regenerate)


def update_state(state, cfg, iiter):
    # This configures the generic interface of `OptimizeVI` for the specific
    # cases of the `linear`, `geometric`, `altmetric` methods.
    regenerate = (
        _getitem(cfg, iiter)['sampling_method'] in ['linear', 'geometric']
    )
    update = (
        _getitem(cfg, iiter)['sampling_method'] in ['geometric', 'altmetric']
    )
    state = state._replace(
        sample_regenerate=regenerate or _do_resample(cfg, iiter),
        sample_update=update,
        kl_solver_kwargs=_getitem(cfg, iiter)['kl_solver_kwargs'],
        sample_generator_kwargs=_getitem(cfg, iiter)['sample_generator_kwargs'],
        sample_update_kwargs=_getitem(cfg, iiter)['sample_update_kwargs'],
    )
    return state


_reduce = partial(tree_map, partial(jnp.mean, axis=0))


def _kl_vg(
    likelihood,
    primals,
    primals_samples,
    *,
    map=jax.vmap,
    reduce=_reduce,
):
    map = get_map(map)

    assert isinstance(primals_samples, Samples)
    ham = StandardHamiltonian(likelihood=likelihood)
    vvg = map(jax.value_and_grad(ham))
    s = vvg(primals_samples.at(primals).samples)
    return reduce(s)


def _kl_met(
    likelihood,
    primals,
    tangents,
    primals_samples,
    *,
    map=jax.vmap,
    reduce=_reduce
):
    map = get_map(map)

    assert isinstance(primals_samples, Samples)
    ham = StandardHamiltonian(likelihood=likelihood)
    vmet = map(ham.metric, in_axes=(0, None))
    s = vmet(primals_samples.at(primals).samples, tangents)
    return reduce(s)


class OptimizeEVIState(NamedTuple):
    nit: int
    key: Array
    samples: Samples
    sample_keys: Array
    sample_instruction: Callable[
        [int],
        Literal[None, "resample_mgvi", "resample_geovi", "curve"],
    ]
    sample_state: OptimizeResults
    minimization_state: OptimizeResults
    kwargs: dict[str, Callable[[int], dict]]


def _make_callable(x):
    if callable(x):
        return x

    def just_return_x(_):
        return x

    return just_return_x


class OptimizeEVI:
    def __init__(
        self,
        likelihood: Likelihood,
        n_total_iterations: int,
        *,
        point_estimates=(),
        constants=(),  # TODO
        kl_jit=True,
        residual_jit=True,
        kl_map=jax.vmap,
        residual_map="lmap",
        kl_reduce=_reduce,
        _kl_value_and_grad: Optional[Callable] = None,
        _kl_metric: Optional[Callable] = None,
        _draw_linear_residual: Optional[Callable] = None,
        _curve_residual: Optional[Callable] = None,
    ):
        """JaxOpt style minimizer for VI approximation of a probability
        distribution with a sampled approximate distribution.

        Parameters:
        -----------
        n_iter: int
            Total number of iterations. One iteration consists of the steps
            1) - 3).
        kl_solver: Callable
            Solver that minimizes the KL w.r.t. the mean of the samples.
        sample_generator: Callable
            Function to generate new samples.
        sample_update: Callable
            Function to update existing samples.
        kl_solver_kwargs: dict
            Optional keyword arguments to be passed on to `kl_solver`. They are
            added to the optimizers state and passed on at each `update` step.
        sample_generator_kwargs: dict
            Optional keyword arguments to be passed on to `sample_generator`.
        sample_update_kwargs: dict
            Optional keyword arguments to be passed on to `sample_update`.

        Notes:
        ------
        Implements the base logic present in conditional VI approximations
        such as MGVI and geoVI. First samples are generated (and/or updated)
        and then their collective mean is optimized for using the sample
        estimated variational KL between the true distribution and the sampled
        approximation. This is split into three steps:
        1) Sample generation
        2) Sample update
        3) KL minimization.
        Step 1) and 2) may be skipped depending on the minimizers state, but
        step 3) is always performed at the end of one iteration. A full loop
        consists of repeatedly iterating over the steps 1) - 3).

        The functions `kl_solver`, `sample_generator`, and `sample_update` all
        share the same syntax: They must take two inputs, samples and keys,
        where keys are the jax.random keys that are used for the samples.
        Additionally they each can take respective keyword arguments. These
        are passed on at runtime and stored in the optimizers state. All
        functions must return samples, as an instance of `Samples`

        TODO:
        MGVI/geoVI interface that creates the input functions of `OptimizeVI`
        from a `Likelihood`.
        Builds functions for a VI approximation via variants of the `Geometric
        Variational Inference` and/or `Metric Gaussian Variational Inference`
        algorithms. They produce approximate posterior samples that are used for KL
        estimation internally and the final set of samples are the approximation of
        the posterior. The samples can be linear, i.e. following a standard normal
        distribution in model space, or non-linear, i.e. following a standard normal
        distribution in the canonical coordinate system of the Riemannian manifold
        associated with the metric of the approximate posterior distribution. The
        coordinate transformation for the non-linear sample is approximated by an
        expansion.
        Both linear and non-linear sample start by drawing a sample from the
        inverse metric. To do so, we draw a sample which has the metric as
        covariance structure and apply the inverse metric to it. The sample
        transformed in this way has the inverse metric as covariance. The first
        part is trivial since we can use the left square root of the metric
        :math:`L` associated with every likelihood:
        .. math::
            \tilde{d} \leftarrow \mathcal{G}(0,\mathbb{1}) \\
            t = L \tilde{d}
        with :math:`t` now having a covariance structure of
        .. math::
            <t t^\dagger> = L <\tilde{d} \tilde{d}^\dagger> L^\dagger = M .
        To transform the sample to an inverse sample, we apply the inverse
        metric. We can do so using the conjugate gradient algorithm (CG). The CG
        algorithm yields the solution to :math:`M s = t`, i.e. applies the
        inverse of :math:`M` to :math:`t`:
        .. math::
            M &s =  t \\
            &s = M^{-1} t = cg(M, t) .
        The linear sample is :math:`s`. The non-linear sample uses :math:`s` as
        a starting value and curves it in a non-linear way as to better resemble
        the posterior locally. See the below reference literature for more
        details on the non-linear sampling.

        Parameters
        ----------
        likelihood : :class:`nifty8.re.likelihood.Likelihood`
            Likelihood to be used for inference.
        n_iter : int
            Number of iterations.
        point_estimates : tree-like structure or tuple of str
            Pytree of same structure as likelihood input but with boolean leaves
            indicating whether to sample the value in the input or use it as a
            point estimate. As a convenience method, for dict-like inputs, a
            tuple of strings is also valid. From these the boolean indicator
            pytree is automatically constructed.
        kl_kwargs: dict
            Keyword arguments passed on to `kl_solver`. Can be used to specify the
            jit and map behavior of the function being constructed.
        linear_sampling_kwargs: dict
            Keyword arguments passed on to `linear_residual_sampler`. Includes
            the cg config used for linear sampling and its jit/map configuration.
        curve_kwargs: dict
            Keyword arguments passed on to `curve_sampler`. Can be used to specify
            the jit and map behavior of the function being constructed.
        _raise_notconverged: bool
            Whether to raise inversion & minimization errors during sampling.
            Default is False.
        See also
        --------
        `Geometric Variational Inference`, Philipp Frank, Reimar Leike,
        Torsten A. Enßlin, `<https://arxiv.org/abs/2105.10470>`_
        `<https://doi.org/10.3390/e23070853>`_
        `Metric Gaussian Variational Inference`, Jakob Knollmüller,
        Torsten A. Enßlin, `<https://arxiv.org/abs/1901.11033>`_
        """
        kl_jit = _parse_jit(kl_jit)
        residual_jit = _parse_jit(residual_jit)
        residual_map = get_map(residual_map)

        if _kl_value_and_grad is None:
            _kl_value_and_grad = kl_jit(
                partial(_kl_vg, likelihood, map=kl_map, reduce=kl_reduce)
            )
        if _kl_metric is None:
            _kl_metric = kl_jit(
                partial(_kl_met, likelihood, map=kl_map, reduce=kl_reduce)
            )
        if _draw_linear_residual is None:
            _draw_linear_residual = residual_jit(
                partial(
                    draw_linear_residual,
                    likelihood,
                    point_estimates=point_estimates,
                )
            )
        if _curve_residual is None:
            # TODO: Pull out `jit` from `curve_residual` once NCG is jit-able
            # TODO: STOP inserting `point_estimes` and instead defer it to `update`
            from .evi import _curve_residual_functions

            _curve_funcs = _curve_residual_functions(
                likelihood=likelihood,
                point_estimates=point_estimates,
                jit=residual_jit,
            )
            _curve_residual = partial(
                curve_residual,
                likelihood,
                point_estimates=point_estimates,
                _curve_funcs=_curve_funcs,
            )

        self.n_total_iterations = None
        self.kl_value_and_grad = None
        self.kl_metric = None
        self.draw_linear_residual = None
        self.curve_residual = None
        self.residual_map = None
        self._replace(
            n_total_iterations=n_total_iterations,
            kl_value_and_grad=_kl_value_and_grad,
            kl_metric=_kl_metric,
            draw_linear_residual=_draw_linear_residual,
            curve_residual=_curve_residual,
            residual_map=residual_map,
        )

    def _replace(
        self,
        *,
        n_total_iterations=None,
        kl_value_and_grad=None,
        kl_metric=None,
        draw_linear_residual=None,
        curve_residual=None,
        residual_map=None,
    ):
        self.n_total_iterations = n_total_iterations if n_total_iterations is None else n_total_iterations
        self.kl_value_and_grad = kl_value_and_grad if kl_value_and_grad is not None else self.kl_value_and_grad
        self.kl_metric = kl_metric if kl_metric is not None else self.kl_metric
        self.draw_linear_residual = draw_linear_residual if draw_linear_residual is not None else self.draw_linear_residual
        self.curve_residual = curve_residual if curve_residual is not None else self.curve_residual
        self.residual_map = residual_map if residual_map is not None else self.residual_map

    def draw_linear_samples(self, primals, keys, **kwargs):
        # NOTE, use `Partial` in favor of `partial` to allow the (potentially)
        # re-jitting `residual_map` to trace the kwargs
        sampler = Partial(self.draw_linear_residual, **kwargs)
        sampler = self.residual_map(sampler, in_axes=(
            None,
            0,
        ))
        smpls, smpls_states = sampler(primals, keys)
        smpls = Samples(
            pos=primals,
            samples=tree_map(lambda *x: jnp.concatenate(x), smpls, -smpls)
        )
        return smpls, smpls_states

    def curve_samples(self, samples, metric_sample_key, **kwargs):
        # NOTE, use `Partial` in favor of `partial` to allow the (potentially)
        # re-jitting `residual_map` to trace the kwargs
        curver = Partial(self.curve_residual, **kwargs)
        curver = self.residual_map(curver, in_axes=(None, 0, 0, 0))
        assert len(metric_sample_key) // 2 == len(samples)
        metric_sample_key = jnp.concatenate((metric_sample_key, ) * 2)
        sgn = jnp.ones(len(samples))
        sgn = jnp.concatenate((sgn, -sgn))
        smpls, smpls_states = curver(
            samples.pos, samples._samples, metric_sample_key, sgn
        )
        return Samples(pos=samples.pos, samples=smpls), smpls_states

    def init_state(
        self,
        nit,
        key,
        *,
        n_samples,
        draw_linear_samples={},
        curve_samples={},
        minimize={"method": "newton-cg"},
        samples=None,
        sample_keys=None,
        sample_instruction=None,
        _kwargs=None
    ):
        if _kwargs is None:
            _kwargs = {
                "n_samples": _make_callable(n_samples),
                "draw_linear_samples": _make_callable(draw_linear_samples),
                "curve_samples": _make_callable(curve_samples),
                "minimize": _make_callable(minimize),
            }
        state = OptimizeEVIState(
            nit,
            key,
            samples=samples,
            sample_keys=sample_keys,
            sample_instruction=sample_instruction,
            kwargs=_kwargs
        )
        return state

    def update(self, params, /, state: OptimizeEVIState, kwargs=None):
        """One sampling and kl optimization step."""
        assert isinstance(samples, Samples)
        assert isinstance(state, OptimizeEVIState)
        nit = state.nit + 1
        key = state.key
        k_smpls = state.sample_keys
        st_smpls = state.sample_state
        samples = state.samples.at(params)
        kwargs = {
            k: _make_callable(v)
            for k, v in kwargs
        } if kwargs is not None else state.kwargs

        smpls_do = state.sample_instruction
        smpls_do = smpls_do(nit) if callable(smpls_do) else smpls_do
        if smpls_do.lower().startswith("resample"):
            n_samples = kwargs["n_samples"](nit)
            key, *k_smpls = random.split(key, n_samples + 1)
            k_smpls = jnp.array(k_smpls)
            kw = kwargs["draw_linear_samples"](nit)
            samples, st_smpls = self.draw_linear_samples(
                samples.pos, k_smpls, **kw
            )
            if smpls_do.lower() == "resample_geovi":
                kw = kwargs["curve_samples"](nit)
                samples, st_smpls = self.curve_samples(samples, k_smpls, **kw)
            elif smpls_do.lower() == "resample_mgvi":
                ve = f"invalid resampling instruction {smpls_do}"
                raise ValueError(ve)
        elif smpls_do.lower() == "curve":
            kw = kwargs["curve_samples"](nit)
            samples, st_smpls = self.curve_samples(samples, k_smpls, **kw)
        elif smpls_do is None:
            ve = f"invalid resampling instruction {smpls_do}"
            raise ValueError(ve)

        # def _minimize_kl(samples, method='newtoncg', method_options={}):
        kw = {
            "fun_and_grad":
                partial(self.kl_value_and_grad, primals_samples=samples),
            "hessp":
                partial(self.kl_metric, primals_samples=samples),
        }
        kw |= kwargs["minimize"](nit)
        kl_opt_state = minimize(None, samples.pos, **kw)
        samples = samples.at(kl_opt_state.x)
        # Remove unnecessary references
        kl_opt_state = kl_opt_state._replace(
            x=None, jac=None, hess=None, hess_inv=None
        )

        state = state._replace(
            niter=nit,
            key=key,
            samples=samples,
            samples_keys=k_smpls,
            sample_state=st_smpls,
            minimization_state=kl_opt_state,
        )
        return samples, state

    def run(self, keys, samples=None, primals=None):
        """`n_total_iterations` consecutive steps of `update`."""
        samples, state = self.init_state(keys, samples, primals)
        for n in range(self.n_total_iterations):
            logger.info(f"OptVI iteration number: {n}")
            samples, state = self.update(samples, state)
        return samples, state


def optimize_kl(
    likelihood: Union[Likelihood, Callable, None],
    pos: Vector,
    total_iterations: int,
    n_samples: Union[int, Callable],
    key: jax.random.PRNGKey,
    point_estimates: Union[Vector, Tuple[str], Callable] = (),
    sampling_method: Union[str, Callable] = 'altmetric',
    make_kl_kwargs: Union[dict, Callable] = {},
    make_sample_generator_kwargs: Union[dict, Callable] = {
        'cg_kwargs': {
            'maxiter': 50
        }
    },
    make_sample_update_kwargs: Union[dict, Callable] = {},
    kl_solver_kwargs: Union[dict, Callable] = {
        'method': 'newtoncg',
        'method_options': {},
    },
    sample_generator_kwargs: dict = {},
    sample_update_kwargs: dict = {
        'minimize_kwargs': {
            'xtol': 0.01
        },
    },
    resample: Union[bool, Callable] = False,
    callback=None,
    out_dir=None,
    resume=False,
    verbosity=0,
    _vi_callables: Union[None, Tuple[Callable], Callable] = None,
    _update_state: Callable = update_state
):
    """Interface for KL minimization similar to NIFTy optimize_kl.

    Parameters
    ----------
    likelihood : :class:`nifty8.re.likelihood.Likelihood` or callable
        Likelihood to be used for inference. If its a callable, must be of the
        form f(current_iteration) -> `Likelihood`. Allows to use different
        likelihoods during minimization.
    pos : Initial position for minimization.
    total_iterations : int
        Number of resampling loops.
    n_samples : int or callable
        Number of samples used to sample Kullback-Leibler divergence. See
        `likelihood` for the callable convention.
    key : jax random number generataion key
    point_estimates : tree-like structure or tuple of str
        Pytree of same structure as `pos` but with boolean leaves indicating
        whether to sample the value in `pos` or use it as a point estimate. As
        a convenience method, for dict-like `pos`, a tuple of strings is also
        valid. From these the boolean indicator pytree is automatically
        constructed.
    sampling_method: str or callable
        Sampling method used for vi approximation. Default is `altmetric`.
    make_kl_kwargs: dict or callable
        Configuration of the KL optimizer passed on to `optimizeVI_callables`.
        Can also be a function of iteration number, in which case
        `optimizeVI_callables` is called again to create new solvers. Note that
        this may trigger re-compilations! The config of the minimizer used in
        the kl optimization can be set at runtime via `kl_solver_kwargs`.
    make_sample_generator_kwargs: dict or callable
        Configuration of the sample generator `linear_sampling` passed on to
        `optimizeVI_callables`. Can also be a function of iteration number.
    make_sample_update_kwargs:  dict or callable
        Configuration of the sample update `curve` passed on to
        `optimizeVI_callables`. Can also be a function of iteration number.
    kl_solver_kwargs: dict or callable
        Keyword arguments to be passed on to `kl_solver` in `OptimizeVI`.
        Specifies the minimizer being used during the kl optimization step and
        its config. Can be a function of iteration number to change the
        minimizers configuration during runtime.
    sample_generator_kwargs: str or callable
        Keyword arguments to be passed on to `sample_generator` in `OptimizeVI`.
        Runtime configuration of the linear sampling.
    sample_update_kwargs: dict or callable
        Keyword arguments to be passed on to `sample_update` in `OptimizeVI`.
        Specifies the minimizer being used during the non-linear `curve` sample
        step and its config.
    resample: bool or callable
        Whether to resample with new random numbers or not. Default is False
    callback : callable or None
        Function that is called after every global iteration. It needs to be a
        function taking 3 arguments: 1. the current samples,
                                     2. the state of `OptimizeVI`,
                                     3. the global iteration number.
        Default: None.
    output_directory : str or None
        Directory in which all output files are saved. If None, no output is
        stored.  Default: None.
    resume : bool
        Resume partially run optimization. If `True` and `output_directory`
        is specified it resumes optimization. Default: False.
    verbosity : int
        Sets verbosity of optimization. If -1 only the current global
        optimization index is printed. If 0 CG steps of linear sampling,
        NewtonCG steps of non linear sampling and NewtonCG steps of KL
        optimization are printed. If set to 1 additionally the internal CG steps
        of the NewtonCG optimization are printed. Default: 0.
    _vi_callables: tuple of callable or callable (optional)
        Option to completely sidestep the `optimizeVI_callables` interface.
        Allows to specify a tuple of the three functions `kl_solver`,
        `sample_generator`, and `sample_update` that are used to instantiate
        `OptimizeVI`. If specified, these functions are used instead of the ones
        created by `optimizeVI_callables` and the corresonding arguments above
        are ignored. Can also be a function of iteration number instead.
    _update_state: callable (Default update_state)
        Function to update the state of `OptimizeVI` according to the config
        specified by the arguments above. The default `update_state` respects
        the MGVI/geoVI logic and implements the corresponding update. If
        `_vi_callables` is set, this may be changed to a different function that
        is applicable to the functions that are being passed on.
    """

    # Prepare dir and load last iteration
    last_finished_index = -1
    if not out_dir == None:
        makedirs(out_dir, exist_ok=True)
        lfile = os.path.join(out_dir, "last_finished_iteration")
        if resume and isfile(lfile):
            with open(lfile) as f:
                last_finished_index = int(f.read())

    # Setup verbosity level
    if verbosity < 0:
        make_sample_generator_kwargs['cg_kwargs']['name'] = None
        kl_solver_kwargs['method_options']['name'] = None
        sample_update_kwargs.get('minimize_kwargs', {})['name'] = None
    else:
        make_sample_generator_kwargs['cg_kwargs'].setdefault(
            'name', 'linear_sampling'
        )
        sample_update_kwargs.get('minimize_kwargs',
                                 {}).setdefault('name', 'non_linear_sampling')
        kl_solver_kwargs['method_options'].setdefault('name', 'minimize')
    if verbosity < 1:
        if "cg_kwargs" in kl_solver_kwargs['method_options'].keys():
            kl_solver_kwargs['method_options']["cg_kwargs"].set_default(
                'name', None
            )
        else:
            kl_solver_kwargs['method_options']["cg_kwargs"] = {"name": None}
        if "cg_kwargs" in sample_update_kwargs['minimize_kwargs'].keys():
            sample_update_kwargs['minimize_kwargs']["cg_kwargs"].set_default(
                'name', None
            )
        else:
            sample_update_kwargs['minimize_kwargs']["cg_kwargs"] = {
                "name": None
            }

    # Split into state changing inputs and constructor inputs of OptimizeVI
    state_cfg = {
        'n_samples': n_samples,
        'sampling_method': sampling_method,
        'resample': resample,
        'kl_solver_kwargs': kl_solver_kwargs,
        'sample_generator_kwargs': sample_generator_kwargs,
        'sample_update_kwargs': sample_update_kwargs,
    }
    constructor_cfg = {
        'likelihood': likelihood,
        'point_estimates': point_estimates,
        'linear_sampling_kwargs': make_sample_generator_kwargs,
        'kl_kwargs': make_kl_kwargs,
        'curve_kwargs': make_sample_update_kwargs,
    }
    # Turn everything into callables by iteration number
    state_cfg = {kk: _make_callable(ii) for kk, ii in state_cfg.items()}
    constructor_cfg = {
        kk: _make_callable(ii)
        for kk, ii in constructor_cfg.items()
    }
    _vi_callables = _make_callable(_vi_callables)

    # Initialize Optimizer
    # If `_vi_callables` are set, use them to set up optimizer directly instead
    # of using `optimizeVI_callables`
    vic = _getitem(_vi_callables, last_finished_index + 1)
    if vic is not None:
        opt = OptimizeVI(n_iter=total_iterations, *vic)
    else:
        kl, lin, geo = optimizeVI_callables(
            **_getitem(constructor_cfg, last_finished_index + 1)
        )
        opt = OptimizeVI(
            n_iter=total_iterations,
            kl_solver=kl,
            sample_generator=lin,
            sample_update=geo
        )

    # Load last finished reconstruction
    if last_finished_index > -1:
        p = os.path.join(out_dir, f"samples_{last_finished_index}.p")
        samples = pickle.load(open(p, "rb"))
        p = os.path.join(out_dir, f"rnd_key_{last_finished_index}.p")
        key = pickle.load(open(p, "rb"))
        p = os.path.join(out_dir, f"state_{last_finished_index}.p")
        state = pickle.load(open(p, "rb"))
        if last_finished_index == total_iterations - 1:
            return samples, state
    else:
        keys = jax.random.split(key, _getitem(state_cfg['n_samples'], 0) + 1)
        key = keys[0]
        samples, state = opt.init_state(keys[1:], primals=pos)
        state = _update_state(state, state_cfg, 0)

    # Update loop
    for i in range(last_finished_index + 1, total_iterations):
        # Do one sampling and minimization step
        samples, state = opt.update(samples, state)
        # Print basic infos
        _optimize_kl_status_print(
            i, samples, state, likelihood.normalized_residual, out_dir=out_dir
        )
        if callback != None:
            callback(samples, state, i)

        if i != total_iterations - 1:
            # Update state
            state = _update_state(state, state_cfg, i + 1)
            if _do_resample(state_cfg, i + 1):
                # Update keys
                keys = jax.random.split(
                    key,
                    _getitem(state_cfg['n_samples'], i + 1) + 1
                )
                key = keys[0]
                state = state._replace(keys=keys[1:])

            # Check for update in constructor and re-initialize sampler
            vic = _getitem(_vi_callables, i)
            if vic is not None:
                if vic != _getitem(_vi_callables, i + 1):
                    opt.set_kl_solver(vic[0])
                    opt.set_sample_generator(vic[1])
                    opt.set_sample_update(vic[2])
            else:
                keep = reduce(
                    lambda a, b: a * b, (
                        _getitem(constructor_cfg[rr], i + 1)
                        == _getitem(constructor_cfg[rr], i)
                        for rr in constructor_cfg.keys()
                    ), True
                )
                if not keep:
                    # TODO print warning
                    # TODO only partial rebuild
                    funcs = optimizeVI_callables(
                        n_iter=total_iterations,
                        **_getitem(constructor_cfg, i + 1)
                    )
                    opt.set_kl_solver(funcs[0])
                    opt.set_sample_generator(funcs[1])
                    opt.set_sample_update(funcs[2])

        if not out_dir == None:
            # TODO: Make this fail safe! Cancelling the run while partially
            # saving the outputs may result in a corrupted state.
            # Save iteration
            p = os.path.join(out_dir, f"rnd_key_{i}.p")
            pickle.dump(key, open(p, "wb"))
            p = os.path.join(out_dir, f"samples_{i}.p")
            pickle.dump(samples, open(p, "wb"))
            p = os.path.join(out_dir, f"state_{i}.p")
            pickle.dump(state, open(p, "wb"))
            p = os.path.join(out_dir, "last_finished_iteration")
            with open(p, "w") as f:
                f.write(str(i))

    return samples, state
