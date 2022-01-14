from jax.config import config

config.update("jax_enable_x64", True)

import sys

from functools import partial
from jax import numpy as jnp
from jax import random
from jax import value_and_grad, jit

import jifty1 as jft


def build_model(predictors, targets, sh, alpha=1):
    my_laplace_prior = jft.interpolate()(jft.laplace_prior(alpha))
    matrix = lambda x: my_laplace_prior(x).reshape(sh)
    model = lambda x: jnp.matmul(predictors, matrix(x))
    lh = jft.Categorical(targets, axis=1)
    return {"lh": lh @ model, "logits": model, "matrix": matrix}


if __name__ == "__main__":
    seed = 42
    key = random.PRNGKey(seed)

    N_data = 1024
    N_categories = 10
    N_predictors = 3

    n_mgvi_iterations = 5
    n_samples = 5
    mirror_samples = True
    n_newton_iterations = 5

    # Create synthetic data
    mock_predictors = random.normal(shape=(N_data, N_predictors), key=key)
    key, subkey = random.split(key)
    model = build_model(
        mock_predictors, jnp.zeros((N_data, 1), dtype=jnp.int32),
        (N_predictors, N_categories)
    )
    latent_truth = random.normal(
        shape=(N_predictors * N_categories, ), key=subkey
    )
    key, subkey = random.split(key)
    matrix_truth = model["matrix"](latent_truth)
    logits_truth = model["logits"](latent_truth)

    mock_targets = random.categorical(logits=logits_truth, key=subkey)
    key, subkey = random.split(key)
    mock_targets = mock_targets.reshape(N_data, 1)

    model = build_model(
        mock_predictors, mock_targets, (N_predictors, N_categories)
    )
    ham = jft.StandardHamiltonian(likelihood=model["lh"]).jit()

    pos_init = .1 * random.normal(
        shape=(N_predictors * N_categories, ), key=subkey
    )
    key, subkey = random.split(key)
    pos = pos_init.copy()

    diff_to_truth = jnp.linalg.norm(model["matrix"](pos) - matrix_truth)
    print(f"Initial diff to truth {diff_to_truth}", file=sys.stderr)

    def energy(p, samps):
        return jnp.mean(jnp.array([ham(p + s) for s in samps]), axis=0)

    energy_vag = jit(value_and_grad(energy))

    @jit
    def metric(p, t, samps):
        results = [ham.metric(p + s, t) for s in samps]
        return jnp.mean(jnp.array(results), axis=0)

    draw = partial(jft.kl.sample_standard_hamiltonian, hamiltonian=ham)

    # Preform MGVI loop
    for i in range(n_mgvi_iterations):
        print(f"MGVI Iteration {i}", file=sys.stderr)
        key, *subkeys = random.split(key, 1 + n_samples)
        samples = []
        samples = [draw(primals=pos, key=k) for k in subkeys]

        Evag = lambda p: energy_vag(p, samples)
        met = lambda p, t: metric(p, t, samples)
        opt_state = jft.minimize(
            None,
            x0=pos,
            method="newton-cg",
            options={
                "fun_and_grad": Evag,
                "hessp": met,
                "maxiter": n_newton_iterations
            }
        )
        pos = opt_state.x
        diff_to_truth = jnp.linalg.norm(model["matrix"](pos) - matrix_truth)
        print(
            (
                f"Post MGVI Iteration {i}: Energy {Evag(pos)[0]:2.4e}"
                f"; diff to truth {diff_to_truth}"
            ),
            file=sys.stderr
        )

    posterior_samps = [s + pos for s in samples]
    import matplotlib.pyplot as plt
    matrix_samps = jnp.array([model["matrix"](s) for s in posterior_samps])
    matrix_mean = jnp.mean(matrix_samps, axis=0)
    matrix_std = jnp.std(matrix_samps, axis=0)
    xx = jnp.linspace(-3.5, 3.5, 2)
    plt.plot(xx, xx)
    plt.errorbar(
        matrix_truth.reshape(-1),
        matrix_mean.reshape(-1),
        yerr=matrix_std.reshape(-1),
        fmt='o',
        color="black"
    )
    plt.xlabel("truth")
    plt.ylabel("inferred value")
    plt.savefig("matrix_fit.png", dpi=400)
    plt.close()