# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

###############################################################################
# FIXME Short text what this demo does
#
#
###############################################################################

import numpy as np

import nifty7 as ift
from matplotlib import pyplot as plt


if __name__ == "__main__":

    # Two-dimensional regular grid with inhomogeneous exposure
    position_space = ift.RGSpace([100])

    # Define harmonic space and harmonic transform
    harmonic_space = position_space.get_default_codomain()
    HT = ift.HarmonicTransformOperator(harmonic_space, position_space)

    # Domain on which the field's degrees of freedom are defined
    domain = ift.DomainTuple.make(harmonic_space)

    # Define amplitude (square root of power spectrum)
    def sqrtpspec(k):
        return 1.0 / (1.0 + k ** 2)

    p_space = ift.PowerSpace(harmonic_space)
    pd = ift.PowerDistributor(harmonic_space, p_space)
    a = ift.PS_field(p_space, sqrtpspec)
    A = pd(a)

    # Define sky operator
    sky = 10 * ift.exp(HT(ift.makeOp(A))).ducktape("xi")

    # M = ift.DiagonalOperator(exposure)
    GR = ift.GeometryRemover(position_space)
    # Define instrumental response
    # R = GR(M)
    R = GR

    # Generate mock data and define likelihood operator
    d_space = R.target[0]
    lamb = R(sky)
    mock_position = ift.from_random(sky.domain, "normal")
    data = lamb(mock_position)
    data = ift.random.current_rng().poisson(data.val.astype(np.float64))
    data = ift.Field.from_raw(d_space, data)
    likelihood = ift.PoissonianEnergy(data) @ lamb

    # Settings for minimization
    ic_newton = ift.DeltaEnergyController(
        name="Newton", iteration_limit=1, tol_rel_deltaE=1e-8
    )

    H = ift.StandardHamiltonian(likelihood)
    position_fc = ift.from_random(H.domain)*0.1
    position_mf = ift.from_random(H.domain)*0.1

    fc = ift.FullCovarianceVI(position_fc, H, 3, True, initial_sig=0.01)
    mf = ift.MeanFieldVI(position_mf, H, 3, True, initial_sig=0.01)
    minimizer_fc = ift.ADVIOptimizer(20, eta = 0.1)
    minimizer_mf = ift.ADVIOptimizer(10)

    plt.pause(0.001)
    for i in range(25):
        if i != 0:
            fc.minimize(minimizer_fc)
            mf.minimize(minimizer_mf)

        plt.figure("result")
        plt.cla()
        plt.plot(
            sky(fc.mean).val,
            "b-",
            label="Full covariance",
        )
        plt.plot(
            sky(mf.mean).val, "r-", label="Mean field"
        )
        for i in range(5):
            plt.plot(
                sky(fc.draw_sample()).val, "b-", alpha=0.3
            )
            plt.plot(
                sky(mf.draw_sample()).val, "r-", alpha=0.3
            )
        plt.plot(data.val, "kx")
        plt.plot(sky(mock_position).val, "k-", label="Ground truth")
        plt.legend()
        plt.ylim(0, data.val.max() + 10)
        plt.pause(0.001)
