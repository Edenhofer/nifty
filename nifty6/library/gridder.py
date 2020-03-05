# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..operators.linear_operator import LinearOperator
from ..sugar import makeField, makeDomain


class GridderMaker(object):
    def __init__(self, dirty_domain, uv, eps=2e-13):
        import nifty_gridder
        dirty_domain = makeDomain(dirty_domain)
        if (len(dirty_domain) != 1 or not isinstance(dirty_domain[0], RGSpace)
                or not len(dirty_domain.shape) == 2):
            raise ValueError("need dirty_domain with exactly one 2D RGSpace")
        if uv.ndim != 2:
            raise ValueError("uv must be a 2D array")
        if uv.shape[1] != 2:
            raise ValueError("second dimension of uv must have length 2")
        dstx, dsty = dirty_domain[0].distances
        # wasteful hack to adjust to shape required by nifty_gridder
        uvw = np.empty((uv.shape[0], 3), dtype=np.float64)
        uvw[:, 0:2] = uv
        uvw[:, 2] = 0.
        # Scale uv such that 0<uv<=1 which is assumed by nifty_gridder
        uvw[:, 0] = uvw[:, 0]*dstx
        uvw[:, 1] = uvw[:, 1]*dsty
        speedOfLight = 299792458.
        bl = nifty_gridder.Baselines(uvw, np.array([speedOfLight]))
        nxdirty, nydirty = dirty_domain.shape
        gconf = nifty_gridder.GridderConfig(nxdirty, nydirty, eps, 1., 1.)
        nu, nv = gconf.Nu(), gconf.Nv()
        self._idx = nifty_gridder.getIndices(
            bl, gconf, np.zeros((uv.shape[0], 1), dtype=np.bool))
        self._bl = bl

        du, dv = 1./(nu*dstx), 1./(nv*dsty)
        grid_domain = RGSpace([nu, nv], distances=[du, dv], harmonic=True)

        self._rest = _RestOperator(dirty_domain, grid_domain, gconf)
        self._gridder = RadioGridder(grid_domain, bl, gconf, self._idx)

    def getGridder(self):
        return self._gridder

    def getRest(self):
        return self._rest

    def getFull(self):
        return self.getRest() @ self._gridder

    def ms2vis(self, x):
        return self._bl.ms2vis(x, self._idx)


class _RestOperator(LinearOperator):
    def __init__(self, dirty_domain, grid_domain, gconf):
        self._domain = makeDomain(grid_domain)
        self._target = makeDomain(dirty_domain)
        self._gconf = gconf
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        res = x.val
        if mode == self.TIMES:
            res = self._gconf.grid2dirty(res)
        else:
            res = self._gconf.dirty2grid(res)
        return makeField(self._tgt(mode), res)


class RadioGridder(LinearOperator):
    def __init__(self, grid_domain, bl, gconf, idx):
        self._domain = DomainTuple.make(
            UnstructuredDomain((bl.Nrows())))
        self._target = DomainTuple.make(grid_domain)
        self._bl = bl
        self._gconf = gconf
        self._idx = idx
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        import nifty_gridder
        self._check_input(x, mode)
        if mode == self.TIMES:
            x = self._bl.ms2vis(x.val.reshape((-1, 1)), self._idx)
            res = nifty_gridder.vis2grid(self._bl, self._gconf, self._idx, x)
        else:
            res = nifty_gridder.grid2vis(self._bl, self._gconf, self._idx,
                                         x.val)
            res = self._bl.vis2ms(res, self._idx).reshape((-1,))
        return makeField(self._tgt(mode), res)
