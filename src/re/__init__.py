# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from .conjugate_gradient import cg, static_cg
from .correlated_field import CorrelatedFieldMaker, non_parametric_amplitude
from .energy_operators import (
    Categorical,
    Gaussian,
    Poissonian,
    StudentT,
    VariableCovarianceGaussian,
    VariableCovarianceStudentT,
)
from .field import Field
from .forest_util import (
    ShapeWithDtype,
    assert_arithmetics,
    dot,
    has_arithmetics,
    norm,
    shape,
    size,
    stack,
    unite,
    unstack,
    vdot,
    vmap_forest,
    vmap_forest_mean,
    zeros_like,
)
from .hmc import generate_hmc_acc_rej, generate_nuts_tree
from .hmc_oo import HMCChain, NUTSChain
from .kl import (
    GeoMetricKL,
    MetricKL,
    geometrically_sample_standard_hamiltonian,
    mean_hessp,
    mean_metric,
    mean_value_and_grad,
    sample_standard_hamiltonian,
)
from .likelihood import Likelihood, StandardHamiltonian
from .optimize import minimize, newton_cg, trust_ncg
from .stats_distributions import (
    invgamma_invprior,
    invgamma_prior,
    laplace_prior,
    lognormal_invprior,
    lognormal_prior,
    normal_prior,
    uniform_prior,
)
from .sugar import (
    ducktape,
    ducktape_left,
    interpolate,
    mean,
    mean_and_std,
    random_like,
    sum_of_squares,
)
