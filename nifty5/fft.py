from __future__ import absolute_import, division, print_function

from .utilities import iscomplextype
import numpy as np


_use_fftw = True


if _use_fftw:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import fftn, rfftn, ifftn
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(1000.)
    # Optional extra arguments for the FFT calls
    # _fft_extra_args = {}
    # if exact reproducibility is needed, use this:
    import os
    nthreads = int(os.getenv("OMP_NUM_THREADS", "1"))
    _fft_extra_args = dict(planner_effort='FFTW_ESTIMATE', threads=nthreads)
else:
    from numpy.fft import fftn, rfftn, ifftn
    _fft_extra_args={}


def hartley(a, axes=None):
    # Check if the axes provided are valid given the shape
    if axes is not None and \
            not all(axis < len(a.shape) for axis in axes):
        raise ValueError("Provided axes do not match array shape")
    if iscomplextype(a.dtype):
        raise TypeError("Hartley transform requires real-valued arrays.")

    tmp = rfftn(a, axes=axes, **_fft_extra_args)

    def _fill_array(tmp, res, axes):
        if axes is None:
            axes = tuple(range(tmp.ndim))
        lastaxis = axes[-1]
        ntmplast = tmp.shape[lastaxis]
        slice1 = (slice(None),)*lastaxis + (slice(0, ntmplast),)
        np.add(tmp.real, tmp.imag, out=res[slice1])

        def _fill_upper_half(tmp, res, axes):
            lastaxis = axes[-1]
            nlast = res.shape[lastaxis]
            ntmplast = tmp.shape[lastaxis]
            nrem = nlast - ntmplast
            slice1 = [slice(None)]*lastaxis + [slice(ntmplast, None)]
            slice2 = [slice(None)]*lastaxis + [slice(nrem, 0, -1)]
            for i in axes[:-1]:
                slice1[i] = slice(1, None)
                slice2[i] = slice(None, 0, -1)
            slice1 = tuple(slice1)
            slice2 = tuple(slice2)
            np.subtract(tmp[slice2].real, tmp[slice2].imag, out=res[slice1])
            for i, ax in enumerate(axes[:-1]):
                dim1 = (slice(None),)*ax + (slice(0, 1),)
                axes2 = axes[:i] + axes[i+1:]
                _fill_upper_half(tmp[dim1], res[dim1], axes2)

        _fill_upper_half(tmp, res, axes)
        return res

    return _fill_array(tmp, np.empty_like(a), axes)


# Do a real-to-complex forward FFT and return the _full_ output array
def my_fftn_r2c(a, axes=None):
    # Check if the axes provided are valid given the shape
    if axes is not None and \
            not all(axis < len(a.shape) for axis in axes):
        raise ValueError("Provided axes do not match array shape")
    if iscomplextype(a.dtype):
        raise TypeError("Transform requires real-valued input arrays.")

    tmp = rfftn(a, axes=axes, **_fft_extra_args)

    def _fill_complex_array(tmp, res, axes):
        if axes is None:
            axes = tuple(range(tmp.ndim))
        lastaxis = axes[-1]
        ntmplast = tmp.shape[lastaxis]
        slice1 = [slice(None)]*lastaxis + [slice(0, ntmplast)]
        res[tuple(slice1)] = tmp

        def _fill_upper_half_complex(tmp, res, axes):
            lastaxis = axes[-1]
            nlast = res.shape[lastaxis]
            ntmplast = tmp.shape[lastaxis]
            nrem = nlast - ntmplast
            slice1 = [slice(None)]*lastaxis + [slice(ntmplast, None)]
            slice2 = [slice(None)]*lastaxis + [slice(nrem, 0, -1)]
            for i in axes[:-1]:
                slice1[i] = slice(1, None)
                slice2[i] = slice(None, 0, -1)
            # np.conjugate(tmp[slice2], out=res[slice1])
            res[tuple(slice1)] = np.conjugate(tmp[tuple(slice2)])
            for i, ax in enumerate(axes[:-1]):
                dim1 = tuple([slice(None)]*ax + [slice(0, 1)])
                axes2 = axes[:i] + axes[i+1:]
                _fill_upper_half_complex(tmp[dim1], res[dim1], axes2)

        _fill_upper_half_complex(tmp, res, axes)
        return res

    return _fill_complex_array(tmp, np.empty_like(a, dtype=tmp.dtype), axes)


def my_fftn(a, axes=None):
    return fftn(a, axes=axes, **_fft_extra_args)
