# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.


_nthreads = 1


def nthreads():
    return _nthreads


def set_nthreads(nthr):
    global _nthreads
    _nthreads = int(nthr)


try:
    import pypocketfft


    def fftn(a, axes=None):
        return pypocketfft.c2c(a, axes=axes, nthreads=max(_nthreads, 0))


    def ifftn(a, axes=None):
        return pypocketfft.c2c(a, axes=axes, inorm=2, forward=False,
                               nthreads=max(_nthreads, 0))


    def hartley(a, axes=None):
        return pypocketfft.genuine_hartley(a, axes=axes,
                                           nthreads=max(_nthreads, 0))

except ImportError:
    import scipy.fft


    def fftn(a, axes=None):
        return scipy.fft.fftn(a, axes=axes, workers=_nthreads)


    def ifftn(a, axes=None):
        return scipy.fft.ifftn(a, axes=axes, workers=_nthreads)


    def hartley(a, axes=None):
        tmp = scipy.fft.fftn(a, axes=axes, workers=_nthreads)
        return tmp.real+tmp.imag
