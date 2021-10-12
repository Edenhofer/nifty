import pytest
import sys
from jax import grad
from jax import numpy as np
from numpy.testing import assert_allclose

import jifty1 as jft

pmp = pytest.mark.parametrize

pot_and_tol = (
    (
        lambda q: np.
        sum(q.T @ np.linalg.inv(np.array([[1, 0.95], [0.95, 1]])) @ q / 2.), 0.2
    ), (lambda q: -1 / np.linalg.norm(q), 2e-2)
)


@pmp("potential_energy, rtol", pot_and_tol)
def test_leapfrog_energy_conservation(potential_energy, rtol):
    dims = (2, )
    mass_matrix = np.ones(shape=dims)
    kinetic_energy = lambda p: np.sum(p**2 / mass_matrix / 2.)

    potential_energy_gradient = grad(potential_energy)
    positions = [np.array([-1.5, -1.55])]
    momenta = [np.array([-1, 1])]
    for _ in range(25):
        new_qp = jft.hmc.leapfrog_step(
            qp=jft.hmc.QP(position=positions[-1], momentum=momenta[-1]),
            potential_energy_gradient=potential_energy_gradient,
            kinetic_energy_gradient=lambda x, y: x * y,
            step_size=0.25,
            inverse_mass_matrix=1. / mass_matrix
        )
        positions.append(new_qp.position)
        momenta.append(new_qp.momentum)

    potential_energies = list(map(potential_energy, positions))
    kinetic_energies = list(map(kinetic_energy, momenta))

    np.set_printoptions(precision=2)
    for q, p, e_kin, e_pot in zip(
        positions, momenta, potential_energies, kinetic_energies
    ):
        msg = (
            f"q: {q}; p: {p}"
            f"\nE_tot: {e_pot+e_kin:.2e}; E_pot: {e_pot:.2e}; E_kin: {e_kin:.2e}"
        )
        print(msg, file=sys.stderr)

    old_energy_tot = potential_energies[0] + kinetic_energies[0]
    new_energy_tot = potential_energies[-1] + kinetic_energies[-1]
    assert_allclose(old_energy_tot, new_energy_tot, rtol=rtol)

    return positions, momenta, kinetic_energies, potential_energies


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    qs, ps, e_kins, e_pots = test_leapfrog_energy_conservation(*pot_and_tol[0])
    positions = np.array(qs)
    momenta = np.array(ps)
    kinetic_energies = np.array(e_kins)
    potential_energies = np.array(e_pots)

    # Position Coordinates
    plt.plot(positions[:, 0], positions[:, 1])
    plt.xlabel("position[:,0]")
    plt.ylabel("position[:,1]")
    plt.show()

    # Momentum coordinates
    plt.plot(momenta[:, 0], momenta[:, 1])
    plt.xlabel("momenta[:,0]")
    plt.ylabel("momenta[:,1]")
    plt.show()

    # Value of Hamiltonian
    # does not look exactly the same as in Neal (2011) unfortunately!
    plt.plot(kinetic_energies, label='kin')
    plt.plot(potential_energies, label='pot')
    plt.plot(kinetic_energies + potential_energies, label='total')
    plt.xlabel('time')
    plt.ylabel('energy')
    plt.legend()
    plt.show()
