import numpy as np

from nifty.config import dependency_injector as gdi
from nifty import HPSpace, LMSpace
from slicing_transformation import SlicingTransformation

import lm_transformation_factory as ltf

hp = gdi.get('healpy')


class HPLMTransformation(SlicingTransformation):

    # ---Overwritten properties and methods---

    def __init__(self, domain, codomain=None, module=None):
        if 'healpy' not in gdi:
            raise ImportError(
                "The module healpy is needed but not available")

        super(HPLMTransformation, self).__init__(domain, codomain, module)

    # ---Mandatory properties and methods---

    @classmethod
    def get_codomain(cls, domain):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  an instance of the :py:class:`lm_space` class.

            Parameters
            ----------
            domain: HPSpace
                Space for which a codomain is to be generated

            Returns
            -------
            codomain : LMSpace
                A compatible codomain.
        """

        if not isinstance(domain, HPSpace):
            raise TypeError(
                "domain needs to be a HPSpace")

        lmax = 3 * domain.nside - 1

        result = LMSpace(lmax=lmax, dtype=np.dtype('float64'))
        cls.check_codomain(domain, result)
        return result

    @staticmethod
    def check_codomain(domain, codomain):
        if not isinstance(domain, HPSpace):
            raise TypeError(
                'ERROR: domain is not a HPSpace')

        if not isinstance(codomain, LMSpace):
            raise TypeError(
                'ERROR: codomain must be a LMSpace.')

        nside = domain.nside
        lmax = codomain.lmax

        if 3 * nside - 1 != lmax:
            raise ValueError(
                'ERROR: codomain has 3*nside-1 != lmax.')

        return None

    def _transformation_of_slice(self, inp, **kwargs):
        lmax = self.codomain.lmax
        mmax = lmax

        if issubclass(inp.dtype.type, np.complexfloating):
            [resultReal, resultImag] = [hp.map2alm(x.astype(np.float64,
                                                            copy=False),
                                                   lmax=lmax,
                                                   mmax=mmax,
                                                   pol=True,
                                                   use_weights=False,
                                                   **kwargs)
                                        for x in (inp.real, inp.imag)]

            [resultReal, resultImag] = [ltf.buildIdx(x, lmax=lmax)
                                        for x in [resultReal, resultImag]]

            result = self._combine_complex_result(resultReal, resultImag)

        else:
            result = hp.map2alm(inp.astype(np.float64, copy=False),
                                lmax=lmax, mmax=mmax, pol=True,
                                use_weights=False)
            result = ltf.buildIdx(result, lmax=lmax)

        return result
