import sys
from datetime import datetime
from functools import partial
from jax import lax
from jax import numpy as np
from jax.tree_util import Partial

from typing import Any, Callable, NamedTuple, Mapping, Optional, Tuple, Union

from . import conjugate_gradient
from .forest_util import size, where
from .forest_util import norm as jft_norm
from .sugar import sum_of_squares


class OptimizeResults(NamedTuple):
    """Object holding optimization results inspired by JAX and scipy.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. NOTE, in contrast to scipy there is no `message` for
        details since strings are not statically memory bound.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    """
    x: Any
    success: Union[bool, np.ndarray]
    status: Union[int, np.ndarray]
    fun: Any
    jac: Any
    hess: Optional[np.ndarray] = None
    hess_inv: Optional[np.ndarray] = None
    nfev: Union[None, int, np.ndarray] = None
    njev: Union[None, int, np.ndarray] = None
    nhev: Union[None, int, np.ndarray] = None
    nit: Union[None, int, np.ndarray] = None
    # Trust-Region specific slots
    trust_radius: Union[None, float, np.ndarray] = None
    jac_magnitude: Union[None, float, np.ndarray] = None
    good_approximation: Union[None, bool, np.ndarray] = None


def _newton_cg(
    fun=None,
    x0=None,
    *,
    maxiter=None,
    energy_reduction_factor=0.1,
    old_fval=None,
    absdelta=None,
    norm_ord=None,
    xtol=1e-5,
    fun_and_grad=None,
    hessp=None,
    cg=conjugate_gradient._cg,
    name=None,
    time_threshold=None,
    cg_kwargs=None
):
    norm_ord = 1 if norm_ord is None else norm_ord
    maxiter = 200 if maxiter is None else maxiter
    xtol = xtol * size(x0)

    pos = x0
    if fun_and_grad is None:
        from jax import value_and_grad

        fun_and_grad = value_and_grad(fun)
    cg_kwargs = {} if cg_kwargs is None else cg_kwargs

    energy, g = fun_and_grad(pos)
    nfev, njev, nhev = 1, 1, 0
    if np.isnan(energy):
        raise ValueError("energy is Nan")
    status = -1
    i = 0
    for i in range(1, maxiter + 1):
        cg_name = name + "CG" if name is not None else None
        # Newton approximates the potential up to second order. The CG energy
        # (`0.5 * x.T @ A @ x - x.T @ b`) and the approximation to the true
        # potential in Newton thus live on comparable energy scales. Hence, the
        # energy in a Newton minimization can be used to set the CG energy
        # convergence criterion.
        if old_fval and energy_reduction_factor:
            cg_absdelta = energy_reduction_factor * (old_fval - energy)
        else:
            cg_absdelta = None if absdelta is None else absdelta / 100.
        mag_g = jft_norm(g, ord=cg_kwargs.get("norm_ord", 1), ravel=True)
        cg_resnorm = np.minimum(0.5, np.sqrt(mag_g)) * mag_g  # taken from SciPy
        default_kwargs = {
            "absdelta": cg_absdelta,
            "resnorm": cg_resnorm,
            "norm_ord": 1,
            "name": cg_name,
            "time_threshold": time_threshold
        }
        cg_res = cg(Partial(hessp, pos), g, **(default_kwargs | cg_kwargs))
        nat_g, info = cg_res.x, cg_res.info
        nhev += cg_res.nfev
        if info is not None and info < 0:
            raise ValueError("conjugate gradient failed")

        naive_ls_it = 0
        dd = nat_g  # negative descent direction
        grad_scaling = 1.
        for naive_ls_it in range(9):
            new_pos = pos - grad_scaling * dd
            new_energy, new_g = fun_and_grad(new_pos)
            nfev, njev = nfev + 1, njev + 1
            if new_energy <= energy:
                break

            grad_scaling /= 2
            if naive_ls_it == 5:
                if name is not None:
                    msg = f"{name}: long line search, resetting"
                    print(msg, file=sys.stderr)
                gam = float(sum_of_squares(g))
                curv = float(g.dot(hessp(pos, g)))
                nhev += 1
                grad_scaling = 1.
                dd = gam / curv * g
        else:
            grad_scaling = 0.
            nm = "N" if name is None else name
            msg = f"{nm}: WARNING: Energy would increase; aborting"
            print(msg, file=sys.stderr)
            status = -1
            break
        if name is not None:
            print(f"{name}: line search: {grad_scaling}", file=sys.stderr)

        if np.isnan(new_energy):
            raise ValueError("energy is NaN")
        energy_diff = energy - new_energy
        old_fval = energy
        energy = new_energy
        pos = new_pos
        g = new_g

        if name is not None:
            msg = f"{name}: Iteration {i} ⛰:{energy:.6e} Δ⛰:{energy_diff:.6e}"
            msg += f" 🞋:{absdelta:.6e}" if absdelta is not None else ""
            print(msg, file=sys.stderr)
        if absdelta is not None and 0. <= energy_diff < absdelta and naive_ls_it < 2:
            status = 0
            break
        if grad_scaling * jft_norm(dd, ord=norm_ord, ravel=True) <= xtol:
            status = 0
            break
        if time_threshold is not None and datetime.now() > time_threshold:
            status = i
            break
    else:
        status = i
        nm = "N" if name is None else name
        print(f"{nm}: Iteration Limit Reached", file=sys.stderr)
    return OptimizeResults(
        x=pos,
        success=True,
        status=status,
        fun=energy,
        jac=g,
        nit=i,
        nfev=nfev,
        njev=njev,
        nhev=nhev
    )


class _TrustRegionState(NamedTuple):
    x: Any
    converged: Union[bool, np.ndarray]
    status: Union[int, np.ndarray]
    fun: Any
    jac: Any
    nfev: Union[int, np.ndarray]
    njev: Union[int, np.ndarray]
    nhev: Union[int, np.ndarray]
    nit: Union[int, np.ndarray]
    trust_radius: Union[float, np.ndarray]
    jac_magnitude: Union[float, np.ndarray]
    good_approximation: Union[bool, np.ndarray]
    old_fval: Union[float, np.ndarray]


def _minimize_trust_ncg(
    fun=None,
    x0: np.ndarray = None,
    *,
    maxiter: Optional[int] = None,
    energy_reduction_factor=0.1,
    old_fval=np.nan,
    absdelta=None,
    norm_ord=None,
    gtol: float = 1e-4,
    max_trust_radius: Union[float, np.ndarray] = 1000.,
    initial_trust_radius: Union[float, np.ndarray] = 1.0,
    eta: Union[float, np.ndarray] = 0.15,
    subproblem=conjugate_gradient._cg_steihaug_subproblem,
    jac: Optional[Callable] = None,
    hessp: Optional[Callable] = None,
    fun_and_grad: Optional[Callable] = None
) -> OptimizeResults:
    norm_ord = 2 if norm_ord is None else norm_ord
    maxiter = 200 if maxiter is None else maxiter

    if not (0 <= eta < 0.25):
        raise Exception("invalid acceptance stringency")
    if gtol < 0.:
        raise Exception("gradient tolerance must be positive")
    if max_trust_radius <= 0:
        raise Exception("max trust radius must be positive")
    if initial_trust_radius <= 0:
        raise ValueError("initial trust radius must be positive")
    if initial_trust_radius >= max_trust_radius:
        ve = "initial trust radius must be less than the max trust radius"
        raise ValueError(ve)

    if fun_and_grad is None:
        from jax import value_and_grad

        fun_and_grad = value_and_grad(fun)
    if hessp is None:
        from jax import grad, jvp

        jac = grad(fun) if jac is None else jac

        def hessp(primals, tangents):
            return jvp(jac, (primals, ), (tangents, ))[1]

    f_0, g_0 = fun_and_grad(x0)

    init_params = _TrustRegionState(
        converged=False,
        status=0,
        good_approximation=np.isfinite(jft_norm(g_0, ord=norm_ord)),
        nit=1,
        x=x0,
        fun=f_0,
        jac=g_0,
        jac_magnitude=jft_norm(g_0, ord=norm_ord),
        nfev=1,
        njev=1,
        nhev=0,
        trust_radius=initial_trust_radius,
        old_fval=old_fval
    )

    def _trust_region_body_f(params: _TrustRegionState) -> _TrustRegionState:
        x_k, g_k, g_k_mag = params.x, params.jac, params.jac_magnitude
        f_k, old_fval = params.fun, params.old_fval
        tr = params.trust_radius

        if energy_reduction_factor:
            cg_absdelta = energy_reduction_factor * (old_fval - f_k)
        else:
            cg_absdelta = None if absdelta is None else absdelta / 100.
        cg_resnorm = np.minimum(0.5, np.sqrt(g_k_mag)) * g_k_mag
        # TODO: add a internal success check for future subproblem approaches
        # that might not be solvable
        result = subproblem(
            f_k,
            g_k,
            partial(hessp, x_k),
            absdelta=cg_absdelta,
            resnorm=cg_resnorm,
            trust_radius=tr,
            norm_ord=norm_ord
        )

        pred_f_kp1 = result.pred_f
        x_kp1 = x_k + result.step
        f_kp1, g_kp1 = fun_and_grad(x_kp1)

        delta = f_k - f_kp1
        pred_delta = f_k - pred_f_kp1

        # update the trust radius according to the actual/predicted ratio
        rho = delta / pred_delta
        cur_tradius = np.where(rho < 0.25, tr * 0.25, tr)
        cur_tradius = np.where(
            (rho > 0.75) & result.hits_boundary,
            np.minimum(2. * tr, max_trust_radius), cur_tradius
        )

        # compute norm to check for convergence
        g_kp1_mag = jft_norm(g_kp1, ord=norm_ord, ravel=True)

        # if the ratio is high enough then accept the proposed step
        f_kp1, x_kp1, g_kp1, g_kp1_mag = where(
            rho > eta, (f_kp1, x_kp1, g_kp1, g_kp1_mag),
            (f_k, x_k, g_k, g_k_mag)
        )
        converged = g_kp1_mag < gtol
        if absdelta:
            energy_diff = f_kp1 - f_k
            converged |= (rho > eta) & (energy_diff >
                                        0.) & (energy_diff < absdelta)

        iter_params = _TrustRegionState(
            converged=converged,
            good_approximation=pred_delta > 0,
            nit=params.nit + 1,
            x=x_kp1,
            fun=f_kp1,
            jac=g_kp1,
            jac_magnitude=g_kp1_mag,
            nfev=params.nfev + result.nfev + 1,
            njev=params.njev + result.njev + 1,
            nhev=params.nhev + result.nhev,
            trust_radius=cur_tradius,
            status=params.status,
            old_fval=f_k
        )

        return iter_params

    def _trust_region_cond_f(params: _TrustRegionState) -> bool:
        return (
            np.logical_not(params.converged) & (params.nit < maxiter) &
            params.good_approximation
        )

    state = lax.while_loop(
        _trust_region_cond_f, _trust_region_body_f, init_params
    )
    status = np.where(
        state.converged,
        0,  # converged
        np.where(
            state.nit == maxiter,
            1,  # max iters reached
            np.where(
                state.good_approximation,
                -1,  # undefined
                2,  # poor approx
            )
        )
    )
    state = state._replace(status=status)

    return OptimizeResults(
        success=state.converged & state.good_approximation,
        nit=state.nit,
        x=state.x,
        fun=state.fun,
        jac=state.jac,
        nfev=state.nfev,
        njev=state.njev,
        nhev=state.nhev,
        jac_magnitude=state.jac_magnitude,
        trust_radius=state.trust_radius,
        status=state.status,
        good_approximation=state.good_approximation
    )


def newton_cg(*args, **kwargs):
    return _newton_cg(*args, **kwargs).x


def trust_ncg(*args, **kwargs):
    return _minimize_trust_ncg(*args, **kwargs).x


def minimize(
    fun: Optional[Callable[..., float]],
    x0,
    args: Tuple = (),
    *,
    method: str,
    tol: Optional[float] = None,
    options: Optional[Mapping[str, Any]] = None
) -> OptimizeResults:
    """Minimize fun."""
    if options is None:
        options = {}
    if not isinstance(args, tuple):
        te = f"args argument must be a tuple, got {type(args)!r}"
        raise TypeError(te)

    fun_with_args = fun
    if args:
        fun_with_args = lambda x: fun(x, *args)

    if tol is not None:
        raise ValueError("use solver-specific options")

    if method.lower() in ('newton-cg', 'newtoncg', 'ncg'):
        return _newton_cg(fun_with_args, x0, **options)
    elif method.lower() in ('trust-ncg', 'trustncg'):
        return _minimize_trust_ncg(fun_with_args, x0, **options)

    raise ValueError(f"method {method} not recognized")
