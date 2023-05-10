#!/usr/bin/env python3

# Copyright(C) 2023 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

# %%
import sys
from functools import partial

import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random
from jax.config import config

import nifty8.re as jft

config.update("jax_enable_x64", True)

dims = (64, 64)
cf_zm = {"offset_mean": 0., "offset_std": (1e-3, 1e-4)}
cf_fl = {
    "fluctuations": (1e-1, 5e-3),
    "loglogavgslope": (-1., 1e-2),
    "flexibility": (4e-1, 2e-1),
    "asperity": (2e-1, 5e-2),
    "harmonic_domain_type": "Fourier"
}
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(**cf_zm)
cfm.add_fluctuations(dims, distances=1. / dims[0], **cf_fl, prefix="ax1")
correlated_field = cfm.finalize()

# %%
seed = 42
key = random.PRNGKey(seed)

signal_response = lambda x: jnp.exp(correlated_field(x))
noise_cov = lambda x: 0.1**2 * x
noise_cov_inv = lambda x: 0.1**-2 * x
noise_std = jnp.sqrt(noise_cov(jnp.ones(correlated_field.target.shape)))

# Create synthetic data
key, subkey = random.split(key)
pos_truth = jft.random_like(subkey, correlated_field.domain)
signal_response_truth = signal_response(pos_truth)
key, subkey = random.split(key)
noise_truth = jnp.sqrt(
    noise_cov(jnp.ones(correlated_field.target.shape))
) * random.normal(shape=correlated_field.target.shape, key=key)
data = signal_response_truth + noise_truth

nll = jft.Gaussian(data, noise_cov_inv) @ signal_response
ham = jft.StandardHamiltonian(likelihood=nll)


def riemannian_manifold_maximum_a_posterior_and_grad(
    pos, data, noise_std, forward, n_samples=1, xmap=jax.vmap
):
    """Riemannian manifold maximum a posteriori and gradient.

    Notes
    -----
    Memory scales quadratically in the number of samples.
    """
    n_eff_samples = 2 * n_samples
    samples_key = random.split(key, n_samples)

    noise_cov_inv = lambda x: x / noise_std**2
    noise_std_inv = lambda x: x / noise_std
    lh_core = partial(
        jft.Gaussian, noise_cov_inv=noise_cov_inv, noise_std_inv=noise_std_inv
    )

    lh = lh_core(data) @ forward
    ham = jft.StandardHamiltonian(lh)

    synth_nll_grad_stack = []  # TODO: pre-allocate stack of samples
    f = forward(pos)  # TODO combine with ham forward pass
    grad_ln_det_metric = jft.zeros_like(pos)
    for k in samples_key:
        d = f + noise_std * random.normal(k, shape=noise_std.shape)
        lh = lh_core(d) @ forward
        tan = jax.grad(lh)(pos)
        synth_nll_grad_stack += [tan]
        grad_ln_det_metric += jft.hvp(lh, (pos, ), (tan, ))

        d = f - noise_std * random.normal(k, shape=noise_std.shape)
        lh = lh_core(d) @ forward
        tan = jax.grad(lh)(pos)
        synth_nll_grad_stack += [tan]
        grad_ln_det_metric += jft.hvp(lh, (pos, ), (tan, ))
    # synth_nll_grad_stack = jft.stack(synth_nll_grad_stack)
    synth_nll_grad_stack = jax.tree_map(
        lambda *x: jnp.stack(tuple(jnp.atleast_1d(el) for el in x)),
        *synth_nll_grad_stack
    )
    small_outer = xmap(xmap(jft.dot, in_axes=(None, 0)), in_axes=(0, None))
    lh_met = 1. / n_eff_samples * small_outer(
        synth_nll_grad_stack, synth_nll_grad_stack
    )
    del synth_nll_grad_stack

    s, ln_det_metric = jnp.linalg.slogdet(jnp.eye(n_eff_samples) + lh_met)
    # assert s == 1
    value, grad = jax.value_and_grad(ham)(pos)
    value += 0.5 * ln_det_metric
    grad += 2 * grad_ln_det_metric / (n_eff_samples * jnp.exp(ln_det_metric))

    return value, grad


# %%
v, g = riemannian_manifold_maximum_a_posterior_and_grad(
    jft.Vector(pos_truth),
    data,
    noise_std,
    forward=signal_response,
    n_samples=2
)
# print(
#     {
#         k: g.tree[k]
#         for k in (
#             "cfax1asperity", "cfax1flexibility", "cfax1fluctuations",
#             "cfax1loglogavgslope", "cfzeromode"
#         )
#     }
# )

# # %%
# print(
#     {
#         k: jax.grad(ham)(jft.Vector(pos_truth)).tree[k]
#         for k in (
#             "cfax1asperity", "cfax1flexibility", "cfax1fluctuations",
#             "cfax1loglogavgslope", "cfzeromode"
#         )
#     }
# )

# %%
n_samples = 4
n_newton_iterations = 25
absdelta = 1e-4 * jnp.prod(jnp.array(dims))

ham_vg = partial(
    riemannian_manifold_maximum_a_posterior_and_grad,
    data=data,
    noise_std=noise_std,
    forward=signal_response,
    n_samples=2
)
ham_metric = jax.jit(ham.metric)

# %%
key, subkey = random.split(key)
pos_init = jft.random_like(subkey, correlated_field.domain)
pos = 1e-2 * jft.Vector(pos_init.copy())

# %%  Minimize the potential
print(f"RMMAP Iteration", file=sys.stderr)
opt_state = jft.minimize(
    None,
    pos,
    method="newton-cg",
    options={
        "fun_and_grad": ham_vg,
        "hessp": ham_metric,
        "absdelta": absdelta,
        "maxiter": n_newton_iterations,
        "name": "N",  # enables verbose logging
    }
)
pos = opt_state.x
msg = f"Post RMMAP Iteration: Energy {ham_vg(pos)[0]:2.4e}"
print(msg, file=sys.stderr)


# %%
@partial(jax.jit, static_argnames=("point_estimates", ))
def sample_evi(primals, key, *, niter, point_estimates=()):
    # at: reset relative position as it gets (wrongly) batched too
    # squeeze: merge "samples" axis with "mirrored_samples" axis
    return jft.smap(
        partial(
            jft.sample_evi,
            nll,
            # linear_sampling_name="S",  # enables verbose logging
            linear_sampling_kwargs={
                "miniter": niter,
                "maxiter": niter
            },
            point_estimates=point_estimates,
        ),
        in_axes=(None, 0)
    )(primals, key).at(primals).squeeze()


samples = sample_evi(pos, random.split(key, 5), niter=50)

# %%
namps = cfm.get_normalized_amplitudes()
post_sr_mean = jft.mean(tuple(signal_response(s) for s in samples.at(pos)))
post_a_mean = jft.mean(tuple(cfm.amplitude(s)[1:] for s in samples.at(pos)))
to_plot = [
    ("Signal", signal_response_truth, "im"),
    ("Noise", noise_truth, "im"),
    ("Data", data, "im"),
    ("Reconstruction", post_sr_mean, "im"),
    ("Ax1", (cfm.amplitude(pos_truth)[1:], post_a_mean), "loglog"),
]
fig, axs = plt.subplots(2, 3, figsize=(16, 9))
for ax, (title, field, tp) in zip(axs.flat, to_plot):
    ax.set_title(title)
    if tp == "im":
        im = ax.imshow(field, cmap="inferno")
        plt.colorbar(im, ax=ax, orientation="horizontal")
    else:
        ax_plot = ax.loglog if tp == "loglog" else ax.plot
        field = field if isinstance(field, (tuple, list)) else (field, )
        for f in field:
            ax_plot(f, alpha=0.7)
fig.tight_layout()
fig.savefig("cf_w_unknown_spectrum.png", dpi=400)
plt.show()
plt.close()
