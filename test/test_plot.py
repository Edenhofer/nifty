# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

import nifty6 as ift


def test_plots():
    # FIXME Write to temporary folder?
    rg_space1 = ift.makeDomain(ift.RGSpace((100,)))
    rg_space2 = ift.makeDomain(ift.RGSpace((80, 60), distances=1))
    hp_space = ift.makeDomain(ift.HPSpace(64))
    gl_space = ift.makeDomain(ift.GLSpace(128))

    fft = ift.FFTOperator(rg_space2)

    field_rg1_1 = ift.Field(rg_space1, np.random.randn(100))
    field_rg1_2 = ift.Field(rg_space1, np.random.randn(100))
    field_rg2 = ift.Field(
        rg_space2, np.random.randn(80*60).reshape((80, 60)))
    field_hp = ift.Field(hp_space, np.random.randn(12*64**2))
    field_gl = ift.Field(gl_space, np.random.randn(32640))
    field_ps = ift.power_analyze(fft.times(field_rg2))

    plot = ift.Plot()
    plot.add(field_rg1_1, title='Single plot')
    plot.output(name='plot1.png')

    plot = ift.Plot()
    plot.add(field_rg2, title='2d rg')
    plot.add([field_rg1_1, field_rg1_2], title='list 1d rg', label=['1', '2'])
    plot.add(field_rg1_2, title='1d rg, xmin, ymin', xmin=0.5, ymin=0.,
             xlabel='xmin=0.5', ylabel='ymin=0')
    plot.output(title='Three plots', name='plot2.png')

    plot = ift.Plot()
    plot.add(field_hp, title='HP planck-color', colormap='Planck-like')
    plot.add(field_rg1_2, title='1d rg')
    plot.add(field_ps)
    plot.add(field_gl, title='GL')
    plot.add(field_rg2, title='2d rg')
    plot.output(nx=2, ny=3, title='Five plots', name='plot3.png')
