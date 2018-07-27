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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

import nifty5 as ift
import numpy as np


def get_random_LOS(n_los):
    starts = list(np.random.uniform(0, 1, (n_los, 2)).T)
    ends = list(np.random.uniform(0, 1, (n_los, 2)).T)
    return starts, ends

class GaussianEnergy2(ift.Operator):
    def __init__(self, mean=None, covariance=None):
        super(GaussianEnergy2, self).__init__()
        self._mean = mean
        self._icov = None if covariance is None else covariance.inverse

    def __call__(self, x):
        residual = x if self._mean is None else x-self._mean
        icovres = residual if self._icov is None else self._icov(residual)
        res = .5 * (residual*icovres).sum()
        metric = ift.SandwichOperator.make(x.jac, self._icov)
        return res.add_metric(metric)

class MyHamiltonian(ift.Operator):
    def __init__(self, lh, ic_samp=None):
        super(MyHamiltonian, self).__init__()
        self._lh = lh
        self._prior = GaussianEnergy2()
        self._ic_samp = ic_samp

    def __call__(self, x):
        res = self._lh(x) + self._prior(x)
        if self._ic_samp is None:
            return self._lh(x) + self._prior(x)
        else:
            lhx = self._lh(x)
            prx = self._prior(x)
            mtr = ift.SamplingEnabler(lhx.metric, prx.metric.inverse,
                                      self._ic_samp, prx.metric.inverse)
            return (lhx+prx).add_metric(mtr)

class EnergyAdapter(ift.Energy):
    def __init__(self, position, op):
        super(EnergyAdapter, self).__init__(position)
        self._op = op
        pvar = ift.Linearization.make_var(position)
        self._res = op(pvar)

    def at(self, position):
        return EnergyAdapter(position, self._op)

    @property
    def value(self):
        return self._res.val.local_data[()]

    @property
    def gradient(self):
        return self._res.gradient

    @property
    def metric(self):
        return self._res.metric

class FieldPicker(ift.Operator):
    def __init__(self, name_dom):
        self._name_dom = name_dom
    def __call__(self, x):
        if isinstance(x, (ift.Field, ift.MultiField)):
            return x[self._name_dom]
        dom = x[self._name_dom].domain
        return ift.Linearization(x._val[self._name_dom], ift.FieldAdapter(dom, self._name_dom, None))

if __name__ == '__main__':
    # FIXME description of the tutorial
    np.random.seed(42)
    position_space = ift.RGSpace([128, 128])

    # Setting up an amplitude model
    A = ift.AmplitudeModel(position_space, 16, 1, 10, -4., 1, 0., 1.)
    dummy = ift.from_random('normal', A.domain)

    # Building the model for a correlated signal
    harmonic_space = position_space.get_default_codomain()
    ht = ift.HarmonicTransformOperator(harmonic_space, position_space)
    power_space = A.target[0]
    power_distributor = ift.PowerDistributor(harmonic_space, power_space)
    dummy = ift.Field.from_random('normal', harmonic_space)

    correlated_field = lambda inp: ht(power_distributor(A(inp))*inp["xi"])
    # alternatively to the block above one can do:
    # correlated_field,_ = ift.make_correlated_field(position_space, A)

    # apply some nonlinearity
    signal = lambda inp:  correlated_field(inp).positive_tanh()

    # Building the Line of Sight response
    LOS_starts, LOS_ends = get_random_LOS(100)
    R = ift.LOSResponse(position_space, starts=LOS_starts,
                        ends=LOS_ends)
    # build signal response model and model likelihood
    signal_response = lambda inp: R(signal(inp))
    # specify noise
    data_space = R.target
    noise = .001
    N = ift.ScalingOperator(noise, data_space)

    # generate mock data
    domain = ift.MultiDomain.union((A.domain, ift.MultiDomain.make({'xi': harmonic_space})))
    MOCK_POSITION = ift.from_random('normal', domain)
    data = signal_response(MOCK_POSITION) + N.draw_sample()

    # set up model likelihood
    likelihood = lambda inp: GaussianEnergy2(mean=data, covariance=N)(signal_response(inp))

    # set up minimization and inversion schemes
    ic_cg = ift.GradientNormController(iteration_limit=10)
    ic_sampling = ift.GradientNormController(iteration_limit=100)
    ic_newton = ift.GradientNormController(name='Newton', iteration_limit=100)
    minimizer = ift.RelaxedNewton(ic_newton)

    # build model Hamiltonian
    H = MyHamiltonian(likelihood, ic_sampling)
    H = EnergyAdapter(MOCK_POSITION, H)

    INITIAL_POSITION = ift.from_random('normal', H.position.domain)
    position = INITIAL_POSITION

    ift.plot(signal(MOCK_POSITION), title='ground truth')
    ift.plot(R.adjoint_times(data), title='data')
    ift.plot([A(MOCK_POSITION)], title='power')
    ift.plot_finish(nx=3, xsize=16, ysize=5, title="setup", name="setup.png")

    # number of samples used to estimate the KL
    N_samples = 20
    for i in range(2):
        H = H.at(position)
        samples = [H.metric.draw_sample(from_inverse=True)
                   for _ in range(N_samples)]

        KL = ift.SampledKullbachLeiblerDivergence(H, samples)
        KL = KL.make_invertible(ic_cg)
        KL, convergence = minimizer(KL)
        position = KL.position

        ift.plot(signal(position), title="reconstruction")

        ift.plot([A(position), A(MOCK_POSITION)],
                 title="power")
        ift.plot_finish(nx=2, xsize=12, ysize=6, title="loop", name="loop.png")

    sc = ift.StatCalculator()
    for sample in samples:
        sc.add(signal(sample+position))
    ift.plot(sc.mean, title="mean")
    ift.plot(ift.sqrt(sc.var), title="std deviation")

    powers = [A(s+position) for s in samples]
    ift.plot([A(position), A(MOCK_POSITION)]+powers,
             title="power")
    ift.plot_finish(nx=3, xsize=16, ysize=5, title="results",
                    name="results.png")
