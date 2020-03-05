# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
import pytest

import nifty6 as ift

from ..common import list2fixture

pmp = pytest.mark.parametrize
space = list2fixture([
    ift.GLSpace(15),
    ift.RGSpace(64, distances=.789),
    ift.RGSpace([32, 32], distances=.789)
])
_h_RG_spaces = [
    ift.RGSpace(7, distances=0.2, harmonic=True),
    ift.RGSpace((12, 46), distances=(.2, .3), harmonic=True)
]
_h_spaces = _h_RG_spaces + [ift.LMSpace(17)]
space1 = space
seed = list2fixture([4, 78, 23])


def testBasics(space, seed):
    np.random.seed(seed)
    S = ift.ScalingOperator(space, 1.)
    s = S.draw_sample()
    var = ift.Linearization.make_var(s)
    model = ift.ScalingOperator(var.target, 6.)
    ift.extra.check_jacobian_consistency(model, var.val)


@pmp('type1', ['Variable', 'Constant'])
@pmp('type2', ['Variable'])
def testBinary(type1, type2, space, seed):
    dom1 = ift.MultiDomain.make({'s1': space})
    dom2 = ift.MultiDomain.make({'s2': space})
    np.random.seed(seed)
    dom = ift.MultiDomain.union((dom1, dom2))
    select_s1 = ift.ducktape(None, dom1, "s1")
    select_s2 = ift.ducktape(None, dom2, "s2")
    model = select_s1*select_s2
    pos = ift.from_random("normal", dom)
    ift.extra.check_jacobian_consistency(model, pos, ntries=20)
    model = select_s1 + select_s2
    pos = ift.from_random("normal", dom)
    ift.extra.check_jacobian_consistency(model, pos, ntries=20)
    model = select_s1.scale(3.)
    pos = ift.from_random("normal", dom1)
    ift.extra.check_jacobian_consistency(model, pos, ntries=20)
    model = ift.ScalingOperator(space, 2.456)(select_s1*select_s2)
    pos = ift.from_random("normal", dom)
    ift.extra.check_jacobian_consistency(model, pos, ntries=20)
    model = ift.sigmoid(2.456*(select_s1*select_s2))
    pos = ift.from_random("normal", dom)
    ift.extra.check_jacobian_consistency(model, pos, ntries=20)
    pos = ift.from_random("normal", dom)
    model = ift.OuterProduct(pos['s1'], ift.makeDomain(space))
    ift.extra.check_jacobian_consistency(model, pos['s2'], ntries=20)
    model = select_s1**2
    pos = ift.from_random("normal", dom1)
    ift.extra.check_jacobian_consistency(model, pos, ntries=20)
    model = select_s1.clip(-1, 1)
    pos = ift.from_random("normal", dom1)
    ift.extra.check_jacobian_consistency(model, pos, ntries=20)
    f = ift.from_random("normal", space)
    model = select_s1.clip(f-0.1, f+1.)
    pos = ift.from_random("normal", dom1)
    ift.extra.check_jacobian_consistency(model, pos, ntries=20)
    if isinstance(space, ift.RGSpace):
        model = ift.FFTOperator(space)(select_s1*select_s2)
        pos = ift.from_random("normal", dom)
        ift.extra.check_jacobian_consistency(model, pos, ntries=20)


def testPointModel(space, seed):
    S = ift.ScalingOperator(space, 1.)
    pos = S.draw_sample()
    alpha = 1.5
    q = 0.73
    model = ift.InverseGammaOperator(space, alpha, q)
    # FIXME All those cdfs and ppfs are not very accurate
    ift.extra.check_jacobian_consistency(model, pos, tol=1e-2, ntries=20)


@pmp('target', [
    ift.RGSpace(64, distances=.789, harmonic=True),
    ift.RGSpace([32, 32], distances=.789, harmonic=True),
    ift.RGSpace([32, 32, 8], distances=.789, harmonic=True)
])
@pmp('causal', [True, False])
@pmp('minimum_phase', [True, False])
@pmp('seed', [4, 78, 23])
def testDynamicModel(target, causal, minimum_phase, seed):
    dct = {
            'target': target,
            'harmonic_padding': None,
            'sm_s0': 3.,
            'sm_x0': 1.,
            'key': 'f',
            'causal': causal,
            'minimum_phase': minimum_phase
            }
    model, _ = ift.dynamic_operator(**dct)
    S = ift.ScalingOperator(model.domain, 1.)
    pos = S.draw_sample()
    # FIXME I dont know why smaller tol fails for 3D example
    ift.extra.check_jacobian_consistency(model, pos, tol=1e-5, ntries=20)
    if len(target.shape) > 1:
        dct = {
            'target': target,
            'harmonic_padding': None,
            'sm_s0': 3.,
            'sm_x0': 1.,
            'key': 'f',
            'lightcone_key': 'c',
            'sigc': 1.,
            'quant': 5,
            'causal': causal,
            'minimum_phase': minimum_phase
        }
        dct['lightcone_key'] = 'c'
        dct['sigc'] = 1.
        dct['quant'] = 5
        model, _ = ift.dynamic_lightcone_operator(**dct)
        S = ift.ScalingOperator(model.domain, 1.)
        pos = S.draw_sample()
        # FIXME I dont know why smaller tol fails for 3D example
        ift.extra.check_jacobian_consistency(
            model, pos, tol=1e-5, ntries=20)


@pmp('h_space', _h_spaces)
@pmp('specialbinbounds', [True, False])
@pmp('logarithmic', [True, False])
@pmp('nbin', [3, None])
def testNormalization(h_space, specialbinbounds, logarithmic, nbin):
    if not specialbinbounds and (not logarithmic or nbin is not None):
        return
    if specialbinbounds:
        binbounds = ift.PowerSpace.useful_binbounds(h_space, logarithmic, nbin)
    else:
        binbounds = None
    dom = ift.PowerSpace(h_space, binbounds)
    op = ift.library.correlated_fields._Normalization(dom)
    pos = 0.1*ift.from_random('normal', op.domain)
    ift.extra.check_jacobian_consistency(op, pos)
