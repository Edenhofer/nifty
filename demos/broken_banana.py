# %%
from jax.config import config

config.update("jax_enable_x64", True)

import sys
import matplotlib.pyplot as plt
from functools import partial
from jax import numpy as np
from jax import lax, random
from jax import value_and_grad, jit

import jifty1 as jft

seed = 42
key = random.PRNGKey(seed)


# %%
def cartesian_product(arrays, out=None):
    import numpy as onp

    # Generalized N-dimensional products
    arrays = [onp.asarray(x) for x in arrays]
    la = len(arrays)
    dtype = onp.find_common_type([a.dtype for a in arrays], [])
    if out is None:
        out = onp.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(onp.ix_(*arrays)):
        out[..., i] = a
    return out.reshape(-1, la)


def banana_helper_phi_b(b, x):
    return np.array([x[0], x[1] + b * x[0]**2 - 100 * b])


def make_kinetic_energy_fn_from_diag_mass_matrix(mass_matrix):
    from jax import tree_util

    def _kin_energy(momentum):
        # calculate kinetic energies for every array (leaf) in the pytree
        kin_energies = tree_util.tree_map(
            lambda p, m: np.sum(p**2 / (2 * m)), momentum, mass_matrix
        )
        # sum everything up
        total_kin_energy = tree_util.tree_reduce(
            lambda acc, leaf_kin_e: acc + leaf_kin_e, kin_energies, 0.
        )
        return total_kin_energy

    return _kin_energy


def make_banana_density(b):
    # https://link.springer.com/article/10.1007/s001800050022
    # b = 0.03 (moderately twisted)
    # b = 0.1 (strongly twisted)
    # kinetic energy is abuse of notation, but gives me a gaussian
    f = make_kinetic_energy_fn_from_diag_mass_matrix(np.array([100, 1]))

    def banana_potential(x):
        return f(banana_helper_phi_b(b, x))

    return banana_potential


# %%
b = 0.1

SCALE = 10.

signal_response = lambda s: banana_helper_phi_b(b, SCALE * s)
nll = jft.Gaussian(
    np.zeros(2), lambda x: x / np.array([100., 1.])
) @ signal_response
nll = nll.jit()
nll_vg = jit(value_and_grad(nll))

ham = jft.StandardHamiltonian(nll)
ham = ham.jit()
ham_vg = jit(value_and_grad(ham))

# %%
n_pix_sqrt = 1000
x = np.linspace(-30 / SCALE, 30 / SCALE, n_pix_sqrt)
y = np.linspace(-65 / SCALE, 15 / SCALE, n_pix_sqrt)
xx = cartesian_product((x, y))
ham_everywhere = np.vectorize(ham, signature="(2)->()")(xx).reshape(
    n_pix_sqrt, n_pix_sqrt
)
plt.imshow(
    np.exp(-ham_everywhere.T),
    extent=(x.min(), x.max(), y.min(), y.max()),
    origin="lower"
)
plt.colorbar()
plt.show()

# %%  # MGVI
n_mgvi_iterations = 30
n_samples = [1] * (n_mgvi_iterations - 1) + [2]
n_newton_iterations = [7] * (n_mgvi_iterations - 10) + [10] * 6 + 4 * [25]
absdelta = 1e-10

initial_position = np.array([1., 1.])
pos = 1e-2 * jft.Field(initial_position)

# Minimize the potential
for i in range(n_mgvi_iterations):
    print(f"MGVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    mkl = jft.MetricKL(
        ham,
        pos,
        n_samples=n_samples[i],
        key=subkey,
        mirror_samples=True,
        hamiltonian_and_gradient=ham_vg,
        #cg_kwargs={"absdelta": absdelta / 10.}
    )

    print("Minimizing...", file=sys.stderr)
    # TODO: Re-introduce a simplified version that works without fields
    opt_state = jft.minimize(
        None,
        x0=pos,
        method="newton-cg",
        options={
            "fun_and_grad": mkl.energy_and_gradient,
            "hessp": mkl.metric,
            "absdelta": absdelta,
            "maxiter": n_newton_iterations[i],
            "name": "N",
        }
    )
    pos = opt_state.x
    print(
        (
            f"Post MGVI Iteration {i}: Energy {ham(pos):2.4e}"
            f"; #NaNs {np.isnan(pos.val).sum()}"
        ),
        file=sys.stderr
    )

mkl_pos = pos

# %%
b_space_smpls = np.array([(mkl_pos + smpl).val for smpl in mkl.samples])

delta = .1
x = np.linspace(-30 / SCALE, 30 / SCALE, n_pix_sqrt)
y = np.linspace(-65 / SCALE, 15 / SCALE, n_pix_sqrt)
X, Y = np.meshgrid(x, y)
XY = np.array([X, Y]).T
xy = XY.reshape((XY.shape[0] * XY.shape[1], 2))
es = np.exp(-lax.map(ham, xy)).reshape(XY.shape[:2]).T

fig, ax = plt.subplots()
contour = ax.contour(X, Y, es)
ax.clabel(contour, inline=True, fontsize=10)
ax.scatter(*b_space_smpls.T)
ax.plot(*mkl_pos, "rx")
plt.show()

# %%  # geoVI
n_geovi_iterations = 15
n_samples = [1] * (n_geovi_iterations - 1) + [2]
n_newton_iterations = [7] * (n_geovi_iterations - 10) + [10] * 6 + [25] * 4
absdelta = 1e-10

initial_position = np.array([1., 1.])
pos = 1e-2 * jft.Field(initial_position)

for i in range(n_geovi_iterations):
    print(f"GeoVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    gkl = jft.GeoMetricKL(
        ham,
        pos,
        n_samples[i],
        key=subkey,
        mirror_samples=True,
        linear_sampling_kwargs={"absdelta": absdelta / 10.},
        non_linear_sampling_kwargs={
            "cg_kwargs": {
                "miniter": 0
            },
            "maxiter": 20,
            "name": "SCG"
        },
        hamiltonian_and_gradient=ham_vg
    )

    print("Minimizing...", file=sys.stderr)
    opt_state = jft.minimize(
        None,
        x0=pos,
        method="newton-cg",
        options={
            "fun_and_grad": gkl.energy_and_gradient,
            "hessp": gkl.metric,
            "absdelta": absdelta,
            "maxiter": n_newton_iterations[i],
            "name": "N",
        }
    )
    pos = opt_state.x

gkl_pos = pos

# %%
b_space_smpls = np.array([(gkl_pos + smpl).val for smpl in gkl.samples])

delta = .1
x = np.linspace(-30 / SCALE, 30 / SCALE, n_pix_sqrt)
y = np.linspace(-65 / SCALE, 15 / SCALE, n_pix_sqrt)
X, Y = np.meshgrid(x, y)
XY = np.array([X, Y]).T
xy = XY.reshape((XY.shape[0] * XY.shape[1], 2))
es = np.exp(-lax.map(ham, xy)).reshape(XY.shape[:2]).T

fig, ax = plt.subplots()
contour = ax.contour(X, Y, es)
ax.clabel(contour, inline=True, fontsize=10)
ax.scatter(*b_space_smpls.T)
ax.plot(*gkl_pos, "rx")
plt.show()
