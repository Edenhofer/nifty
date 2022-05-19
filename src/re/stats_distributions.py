from typing import Callable, Optional

from jax import numpy as jnp


def laplace_prior(alpha) -> Callable:
    """
    Takes random normal samples and outputs samples distributed according to

    .. math::
        P(x|a) = exp(-|x|/a)/a/2

    """
    from jax.scipy.stats import norm

    def standard_to_laplace(xi):
        res = (xi < 0) * (norm.logcdf(xi) + jnp.log(2))
        res -= (xi > 0) * (norm.logcdf(-xi) + jnp.log(2))
        return res * alpha

    return standard_to_laplace


def normal_prior(mean, std) -> Callable:
    """Match standard normally distributed random variables to non-standard
    variables.
    """
    def standard_to_normal(xi):
        return mean + std * xi

    return standard_to_normal


def lognormal_moments(mean, std):
    """Compute the cumulants a log-normal process would need to comply with the
    provided mean and standard-deviation `std`
    """
    if jnp.any(mean <= 0.):
        raise ValueError(f"`mean` must be greater zero; got {mean!r}")
    if jnp.any(std <= 0.):
        raise ValueError(f"`std` must be greater zero; got {std!r}")

    logstd = jnp.sqrt(jnp.log1p((std / mean)**2))
    logmean = jnp.log(mean) - 0.5 * logstd**2
    return logmean, logstd


def lognormal_prior(mean, std) -> Callable:
    """Moment-match standard normally distributed random variables to log-space

    Takes random normal samples and outputs samples distributed according to

    .. math::
        P(xi|mu,sigma) \\propto exp(mu + sigma * xi)

    such that the mean and standard deviation of the distribution matches the
    specified values.
    """
    standard_to_normal = normal_prior(*lognormal_moments(mean, std))

    def standard_to_lognormal(xi):
        return jnp.exp(standard_to_normal(xi))

    return standard_to_lognormal


def lognormal_invprior(mean, std) -> Callable:
    """Get the inverse transform to `lognormal_prior`."""
    ln_m, ln_std = lognormal_moments(mean, std)

    def lognormal_to_standard(y):
        return (jnp.log(y) - ln_m) / ln_std

    return lognormal_to_standard


def uniform_prior(a_min=0., a_max=1.) -> Callable:
    """Transform a standard normal into a uniform distribution.

    Parameters
    ----------
    a_min : float
        Minimum value.
    a_max : float
        Maximum value.
    """
    from jax.scipy.stats import norm

    if a_min == 0. and a_max == 1.:
        return norm.cdf

    scale = a_max - a_min

    def standard_to_uniform(xi):
        return a_min + scale * norm.cdf(xi)

    return standard_to_uniform


def interpolator(
    func: Callable,
    xmin: float,
    xmax: float,
    *,
    step: Optional[float] = None,
    num: Optional[int] = None,
    table_func: Optional[Callable] = None,
    inv_table_func: Optional[Callable] = None,
    return_inverse: Optional[bool] = False
):  # Adapted from NIFTy
    """
    Evaluate a function point-wise by interpolation.  Can be supplied with a
    table_func to increase the interpolation accuracy, Best results are
    achieved when `lambda x: table_func(func(x))` is roughly linear.

    Parameters
    ----------
    func : function
        Function to interpolate.
    xmin : float
        The smallest value for which `func` will be evaluated.
    xmax : float
        The largest value for which `func` will be evaluated.
    step : float
        Distance between sampling points for linear interpolation. Either of
        `step` or `num` must be specified.
    num : int
        The number of interpolation points. Either of `step` of `num` must be
        specified.
    table_func : function
        Non-linear function applied to the tabulated function in order to
        transform the table to a more linear space.
    inv_table_func : function
        Inverse of `table_func`.
    return_inverse : bool
        Whether to also return the interpolation of the inverse of `func`. Only
        sensible if `func` is invertible.
    """
    # from scipy.interpolate import CubicSpline

    if step is not None and num is not None:
        ve = "either but not both of `step` and `num` must be specified"
        raise ValueError(ve)
    if step is not None:
        xs = jnp.arange(xmin, xmax + step, step)
    elif num is not None:
        xs = jnp.linspace(xmin, xmax, num)
    else:
        ve = "either of `step` or `num` must be specified"
        raise ValueError(ve)

    ys = func(xs)
    if table_func is not None:
        if inv_table_func is None:
            raise ValueError("no `inv_table_func` specified")
        ys = table_func(ys)

    # interpolator = CubicSpline(xs, ys)
    # deriv = interpolator.derivative()

    def interp(x):
        # res = interpolator(x)
        res = jnp.interp(x, xs, ys)
        if inv_table_func is not None:
            res = inv_table_func(res)
        return res

    if return_inverse:

        def inverse_interp(y):
            if table_func is not None:
                y = table_func(y)
            return jnp.interp(y, ys, xs)

        return interp, inverse_interp

    return interp


def invgamma_prior(a, scale, loc=0., step=1e-2) -> Callable:
    """Transform a standard normal into an inverse gamma distribution.

    The pdf of the inverse gamma distribution is defined as follows using
    :math:`q` to denote the scale:

    .. math::

        P(x|q, a) = \\frac{q^a}{\\Gamma(a)}x^{-a -1}
        \\exp \\left(-\\frac{q}{x}\\right)

    That means that for large x the pdf falls off like :math:`x^{(-a -1)}`.
    The mean of the pdf is at :math:`q / (a - 1)` if :math:`a > 1`.
    The mode is :math:`q / (a + 1)`.

    This transformation is implemented as a linear interpolation which maps a
    Gaussian onto an inverse gamma distribution.

    Parameters
    ----------
    a : float
        The shape-parameter of the inverse-gamma distribution.
    scale : float
        The scale-parameter of the inverse-gamma distribution.
    loc : float
        An option shift of the whole distribution.
    step : float
        Distance between sampling points for linear interpolation.
    """
    from scipy.stats import invgamma, norm

    if not jnp.isscalar(a) or not jnp.isscalar(loc):
        te = (
            "Shape `a` and location `loc` must be of scalar type"
            f"; got {type(a)} and {type(loc)} respectively"
        )
        raise TypeError(te)
    if loc == 0.:
        # Pull out `scale` to interpolate less
        s2i = lambda x: invgamma.ppf(norm._cdf(x), a=a)
    elif jnp.isscalar(scale):
        s2i = lambda x: invgamma.ppf(norm._cdf(x), a=a, loc=loc, scale=scale)
    else:
        raise TypeError("`scale` may only be array-like for `loc == 0.`")

    xmin, xmax = -8.2, 8.2  # (1. - norm.cdf(8.2)) * 2 < 1e-15
    standard_to_invgamma_interp = interpolator(
        s2i, xmin, xmax, step=step, table_func=jnp.log, inv_table_func=jnp.exp
    )

    def standard_to_invgamma(x):
        # Allow for array-like `scale` without separate interpolations and only
        # interpolate for shape `a` and `loc`
        if loc == 0.:
            return standard_to_invgamma_interp(x) * scale
        return standard_to_invgamma_interp(x)

    return standard_to_invgamma


def invgamma_invprior(a, scale, loc=0., step=1e-2) -> Callable:
    """Get the inverse transformation to `invgamma_prior`."""
    from scipy.stats import invgamma, norm

    xmin, xmax = -8.2, 8.2  # (1. - norm.cdf(8.2)) * 2 < 1e-15
    _, invgamma_to_standard = interpolator(
        lambda x: invgamma.ppf(norm._cdf(x), a=a, loc=loc, scale=scale),
        xmin,
        xmax,
        step=step,
        table_func=jnp.log,
        inv_table_func=jnp.exp,
        return_inverse=True
    )
    return invgamma_to_standard
