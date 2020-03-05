# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_equal

import nifty6 as ift

pmp = pytest.mark.parametrize
# [shape, distances, harmonic, expected]
CONSTRUCTOR_CONFIGS = [[(8,), None, False, {
    'shape': (8,),
    'distances': (0.125,),
    'harmonic': False,
    'size': 8,
}], [(8,), None, True, {
    'shape': (8,),
    'distances': (1.0,),
    'harmonic': True,
    'size': 8,
}], [(8,), (12,), True, {
    'shape': (8,),
    'distances': (12.0,),
    'harmonic': True,
    'size': 8,
}], [(11, 11), None, False, {
    'shape': (11, 11),
    'distances': (1/11, 1/11),
    'harmonic': False,
    'size': 121,
}], [(12, 12), (1.3, 1.3), True, {
    'shape': (12, 12),
    'distances': (1.3, 1.3),
    'harmonic': True,
    'size': 144,
}]]


def get_k_length_array_configs():
    # for RGSpace(shape=(4, 4), distances=(0.25,0.25))
    cords_0 = np.ogrid[0:4, 0:4]
    da_0 = ((cords_0[0] - 4//2)*0.25)**2
    da_0 = np.fft.ifftshift(da_0)
    temp = ((cords_0[1] - 4//2)*0.25)**2
    temp = np.fft.ifftshift(temp)
    da_0 = da_0 + temp
    da_0 = np.sqrt(da_0)
    return [
        [(4, 4), (0.25, 0.25), da_0],
    ]


def get_dvol_configs():
    np.random.seed(42)
    return [[(11, 11), None, False, 1], [(11, 11), None, False, 1],
            [(11, 11), (1.3, 1.3), False, 1], [(11, 11), (1.3, 1.3), False,
                                               1], [(11, 11), None, True, 1],
            [(11, 11), None, True, 1], [(11, 11), (1.3, 1.3), True,
                                        1], [(11, 11), (1.3, 1.3), True, 1]]


@pmp('attribute, expected_type', [['distances', tuple]])
def test_property_ret_type(attribute, expected_type):
    x = ift.RGSpace(1)
    assert_(isinstance(getattr(x, attribute), expected_type))


@pmp('shape, distances, harmonic, expected', CONSTRUCTOR_CONFIGS)
def test_constructor(shape, distances, harmonic, expected):
    x = ift.RGSpace(shape, distances, harmonic)
    for key, value in expected.items():
        assert_equal(getattr(x, key), value)


@pmp('shape, distances, expected', get_k_length_array_configs())
def test_k_length_array(shape, distances, expected):
    r = ift.RGSpace(shape=shape, distances=distances, harmonic=True)
    assert_allclose(r.get_k_length_array().val, expected)


@pmp('shape, distances, harmonic, power', get_dvol_configs())
def test_dvol(shape, distances, harmonic, power):
    r = ift.RGSpace(shape=shape, distances=distances, harmonic=harmonic)
    assert_allclose(r.dvol, np.prod(r.distances)**power)
