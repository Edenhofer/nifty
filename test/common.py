# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import pytest


def list2fixture(lst):
    @pytest.fixture(params=lst)
    def myfixture(request):
        return request.param

    return myfixture
