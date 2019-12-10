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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import scipy.fft

_nthreads = 1


def nthreads():
    return _nthreads


def set_nthreads(nthr):
    global _nthreads
    _nthreads = int(nthr)


def fftn(a, axes=None):
    return scipy.fft.fftn(a, axes=axes, workers=_nthreads)


def ifftn(a, axes=None):
    return scipy.fft.ifftn(a, axes=axes, workers=_nthreads)


def hartley(a, axes=None):
    tmp = scipy.fft.fftn(a, axes=axes, workers=_nthreads)
    return tmp.real+tmp.imag
