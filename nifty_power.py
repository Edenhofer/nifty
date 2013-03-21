## NIFTY (Numerical Information Field Theory) has been developed at the
## Max-Planck-Institute for Astrophysics.
##
## Copyright (C) 2013 Max-Planck-Society
##
## Author: Marco Selig
## Project homepage: <http://www.mpa-garching.mpg.de/ift/nifty/>
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
## See the GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
    ..                     __   ____   __
    ..                   /__/ /   _/ /  /_
    ..         __ ___    __  /  /_  /   _/  __   __
    ..       /   _   | /  / /   _/ /  /   /  / /  /
    ..      /  / /  / /  / /  /   /  /_  /  /_/  /
    ..     /__/ /__/ /__/ /__/    \___/  \___   /  power
    ..                                  /______/

    NIFTy offers a number of automatized routines for handling
    power spectra. It is possible to draw a field from a random distribution
    with a certain autocorrelation or, equivalently, with a certain
    power spectrum in its conjugate space (see :py:func:`field.random`). In
    NIFTy, it is usually assumed that such a field follows statistical
    homogeneity and isotropy. Fields which are only statistically homogeneous
    can also be created using the diagonal operator routine.

    At the moment, NIFTY offers several additional routines for power spectrum
    manipulation.

"""

from __future__ import division
from nifty_core import *
#import numpy as np
import smoothing as gs


##-----------------------------------------------------------------------------

def weight_power(domain,spec,power=1,pindex=None,pundex=None):
    """
        Weights a given power spectrum with the corresponding pixel volumes
        to a given power.

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.

        spec : {scalar, ndarray, field}
            The power spectrum. A scalars is interpreted as a constant
            spectrum.
        pindex : ndarray
            Indexing array giving the power spectrum index for each
            represented mode.
        pundex : list
            Unindexing list undoing power indexing.

        Returns
        -------
        spev : ndarray
            Weighted power spectrum.

        Raises
        ------
        TypeError
            If `domain` is no space.
        ValueError
            If `domain` is no harmonic space.

    """
    ## check domain
    if(not isinstance(domain,space)):
        raise TypeError(about._errors.cstring("ERROR: invalid input."))

    if(pindex is None):
        try:
            pindex = domain.get_power_index(irreducible=False)
        except(AttributeError):
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
    if(pundex is None):
        pundex = domain.get_power_undex(pindex=pindex)

    return np.real(domain.calc_weight(domain.enforce_power(spec,size=len(set(pindex.flatten(order='C'))))[pindex],power=power)[pundex])

##-----------------------------------------------------------------------------

##-----------------------------------------------------------------------------

def smooth_power(power,kindex,mode="2s",exclude=1,sigma=-1):
    """
    Smoothes a power spectrum via convolution with a Gaussian kernel.

    Parameters
    ----------
    power : ndarray
        The power spectrum to be smoothed.

    kindex : ndarray
        The array specifying the coordinate indices in conjugate space.

    mode : string
        Specifies the smoothing mode (default: "2s") :

        - "ff" (smoothing in the harmonic basis using fast Fourier transformations)
        - "bf" (smoothing in the position basis by brute force)
        - "2s" (smoothing in the position basis restricted to a 2-`sigma` interval)

    exclude : scalar
        Excludes the first power spectrum entries from smoothing, indicated by
        the given integer scalar (default = 1, the monopol is not smoothed).

    sigma : scalar
        FWHM of Gaussian convolution kernel (default = -1, `sigma` is set
        automatically).

    Returns
    -------
    smoothpower : ndarray
        The smoothed power spectrum.

    Raises
    ------
    KeyError
        If `mode` is unsupported.

    """
    if(mode=="2s"):
        return gs.smooth_power_2s(power,kindex,exclude=exclude,smooth_length=sigma)
    elif(mode=="ff"):
        return gs.smooth_power(power,kindex,exclude=exclude,smooth_length=sigma)
    elif(mode=="bf"):
        return gs.smooth_power_bf(power,kindex,exclude=exclude,smooth_length=sigma)
    else:
        raise KeyError(about._errors.cstring("ERROR: unsupported mode '"+str(mode)+"'."))

##-----------------------------------------------------------------------------

##=============================================================================

def _calc_laplace(kindex): ## > computes Laplace operator and integrand
    ## finite differences
    l = np.r_[0,0,np.log(kindex[2:]/kindex[1])]
    dl1 = l[1:]-l[:-1]
    dl2 = l[2:]-l[:-2]
    if(np.any(dl1[1:]==0))or(np.any(dl2==0)):
        raise ValueError(about._errors.cstring("ERROR: too finely divided harmonic grid."))
    ## operator(s)
    klim = len(kindex)
    L = np.zeros((klim,klim))
    I = np.zeros(klim)
    for jj in range(2,klim-1): ## leave out {0,1,kmax}
        L[jj,jj-1] = 2/(dl2[jj-1]*dl1[jj-1])
        L[jj,jj] = -2/dl2[jj-1]*(1/dl1[jj]+1/dl1[jj-1])
        L[jj,jj+1] = 2/(dl2[jj-1]*dl1[jj])
        I[jj] = dl2[jj-1]/2
    return L,I

def _calc_inverse(tk,var,kindex,rho,b1,Amem): ## > computes the inverse Hessian `A` and `b2`
    ## operator `T` from Eq.(B8) times 2
    if(Amem is None):
        L,I = _calc_laplace(kindex)
        #T2 = 2*np.dot(L.T,np.dot(np.diag(I/var,k=0),L,out=None),out=None) # Eq.(B8) * 2
        if(np.isscalar(var)):
            Amem = np.dot(L.T,np.dot(np.diag(I,k=0),L,out=None),out=None)
            T2 = 2/var*Amem
        else:
            Amem = np.dot(np.diag(np.sqrt(I),k=0),L,out=None)
            T2 = 2*np.dot(Amem.T,np.dot(np.diag(1/var,k=0),Amem,out=None),out=None)
    elif(np.isscalar(var)):
        T2 = 2/var*Amem
    else:
        T2 = 2*np.dot(Amem.T,np.dot(np.diag(1/var,k=0),Amem,out=None),out=None)
    b2 = b1+np.dot(T2,tk,out=None)
    ## inversion
    return np.linalg.inv(T2+np.diag(b2,k=0)),b2,Amem

def infer_power(m,domain=None,Sk=None,D=None,pindex=None,pundex=None,kindex=None,rho=None,q=1E-42,alpha=1,perception=(1,0),smoothness=False,var=100,bare=True,**kwargs):
    """
        Infers the power spectrum.

        Given a map the inferred power spectrum is equal to ``m.power()``; given
        an uncertainty a power spectrum is inferred according to the "critical"
        filter formula, which can be extended by a smoothness prior. For
        details, see references below.

        Parameters
        ----------
        m : field
            Map for which the power spectrum is inferred.
        domain : space
            The space wherein the power spectrum is defined, can be retrieved
            from `Sk.domain` (default: None).
        Sk : projection_operator
            Projection operator specifying the pseudo trace for all projection
            bands, can be initialized from `domain` and `pindex`
            (default: None).
        D : operator, *optional*
            Operator expressing the uncertainty of the map `m`, its diagonal
            `D.hathat` in the `domain` suffices (default: 0).
        pindex : numpy.ndarray, *optional*
            Indexing array giving the power spectrum index for each
            represented mode (default: None).
        pundex : list, *optional*
            Unindexing list undoing power indexing.
        kindex : numpy.ndarray, *optional*
            Scale corresponding to each band in the power spectrum
            (default: None).
        rho : numpy.ndarray, *optional*
            Number of modes per scale (default: None).
        q : {scalar, list, array}, *optional*
            Spectral scale parameter of the assumed inverse-Gamme prior
            (default: 1E-42).
        alpha : {scalar, list, array}, *optional*
            Spectral shape parameter of the assumed inverse-Gamme prior
            (default: 1).
        perception : {tuple, list, array}, *optional*
            Tuple specifying the filter perception (delta,epsilon)
            (default: (1,0)).
        smoothness : bool, *optional*
            Indicates whether the smoothness prior is used or not
            (default: False).
        var : {scalar, list, array}, *optional*
            Variance of the assumed spectral smoothness prior (default: 100).
        bare : bool, *optional*
            Indicates whether the power spectrum entries returned are "bare"
            or not (mandatory for the correct incorporation of volume weights)
            (default: True).

        Returns
        -------
        pk : numpy.ndarray
            The inferred power spectrum, weighted according to the `bare` flag.

        Other Parameters
        ----------------
        random : string, *optional*
            The distribution from which the probes for the diagonal probing are
            drawn, supported distributions are (default: "pm1"):

            - "pm1" (uniform distribution over {+1,-1} or {+1,+i,-1,-i})
            - "gau" (normal distribution with zero-mean and unit-variance)

        ncpu : int, *optional*
            The number of CPUs to be used for parallel probing (default: 2).
        nrun : int, *optional*
            The number of probes to be evaluated; if ``nrun < ncpu ** 2``, it
            will be set to ``ncpu ** 2`` (default: 8).
        nper : int, *optional*
            This number specifies how many probes will be evaluated by one
            worker. Afterwards a new worker will be created to evaluate a chunk
            of `nper` probes; it is recommended to stay with the default value
            (default: None).
        save : bool, *optional*
            If `save` is True, then the probing results will be written to the
            hard disk instead of being saved in the RAM; this is recommended
            for high dimensional fields whose probes would otherwise fill up
            the memory (default: False).
        path : string, *optional*
            The path, where the probing results are saved, if `save` is True
            (default: "tmp").
        prefix : string, *optional*
            A prefix for the saved probing results; the saved results will be
            named using that prefix and an 8-digit number (default: "").

        Notes
        -----
        The general approach to inference of unknown power spectra is detailed
        in [#]_, where the "critical" filter formula, Eq.(37b), used here is
        derived, and the implications of a certain choise of the perception
        tuple (delta,epsilon) are discussed.
        The further incorporation of a smoothness prior as detailed in [#]_,
        where the underlying formula(s), Eq.(27), of this implementation are
        derived and discussed in terms of their applicability.

        References
        ----------
        .. [#] T.A. Ensslin and M. Frommert, "Reconstruction of signals with
            unknown spectra in information field theory with parameter
            uncertainty", Physical Review E, 2011,
            10.1103/PhysRevD.83.105014;
            `arXiv:1002.2928 <http://www.arxiv.org/abs/1002.2928>`_
        .. [#] N. Opermann et. al., "Reconstruction of Gaussian and log-normal
            fields with spectral smoothness", Physical Review E, 2013,
            10.1103/PhysRevE.87.032136;
            `arXiv:1210.6866 <http://www.arxiv.org/abs/1210.6866>`_

        Raises
        ------
        IndexError, TypeError, ValueError
            If some input is invalid.

    """
    ## check map
    if(not isinstance(m,field)):
        raise TypeError(about._errors.cstring("ERROR: invalid input."))
    ## check domain
    if(domain is None):
        if(Sk is None):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        else:
            domain = Sk.domain
    elif(not isinstance(domain,space)):
        raise TypeError(about._errors.cstring("ERROR: invalid input."))
    ## check power indices
    if(pindex is None):
        try:
            pindex = domain.get_power_index(irreducible=False)
        except(AttributeError):
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
    else:
        pindex = np.array(pindex,dtype=np.int)
        if(not np.all(np.array(np.shape(pindex))==domain.dim(split=True))):
            raise ValueError(about._errors.cstring("ERROR: shape mismatch ( "+str(np.array(np.shape(pindex)))+" <> "+str(domain.dim(split=True))+" )."))
    if(pundex is None):
        pundex = domain.get_power_undex(pindex=pindex)
    if(kindex is None)or(rho is None):
        try:
            kindex,rho = domain.get_power_index(irreducible=True)
        except(AttributeError):
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
    ## check projection operator
    if(Sk is None):
        Sk = projection_operator(domain,assign=pindex)
    elif(not isinstance(Sk,projection_operator))or(not hasattr(Sk,"pseudo_tr")):
        raise TypeError(about._errors.cstring("ERROR: invalid input."))
    elif(Sk.domain<>domain):
        raise ValueError(about._errors.cstring("ERROR: invalid input."))
    ## check critical parameters
    if(not np.isscalar(q)):
        q = np.array(q,dtype=domain.vol.dtype).flatten()
        if(np.size(q)<>np.size(kindex)):
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
    if(not np.isscalar(alpha)):
        alpha = np.array(alpha,dtype=domain.vol.dtype).flatten()
        if(np.size(alpha)<>np.size(kindex)):
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
    ## check perception (delta,epsilon)
    if(perception is None):
        perception = (1,0) ## critical perception
    elif(not isinstance(perception,(tuple,list,np.ndarray))):
        raise TypeError(about._errors.cstring("ERROR: invalid input."))
    elif(len(perception)<2):
        raise IndexError(about._errors.cstring("ERROR: invalid input."))
    if(perception[1] is None):
        perception[1] = rho/2*(perception[0]-1) ## critical epsilon
    ## check smothness variance
    if(not np.isscalar(var)):
        var = np.array(var,dtype=domain.vol.dtype).flatten()
        if(np.size(var)<>np.size(kindex)):
            raise ValueError(about._errors.cstring("ERROR: invalid input."))

    ## trace(s) of B
    trB1 = Sk.pseudo_tr(m) ## == Sk(m).pseudo_dot(m), but faster
    if(perception[0]==0)or(D is None)or(D==0):
        trB2 = 0
    else:
        trB2 = Sk.pseudo_tr(D,**kwargs) ## probing of the partial traces of D
    ## power spectrum
    numerator = 2*q+trB1+perception[0]*trB2 ## non-bare(!)
    denominator1 = rho+2*(alpha-1+perception[1])

    if(smoothness):
        if(not domain.discrete):
            numerator = weight_power(domain,numerator,power=-1,pindex=pindex,pundex=pundex)
        pk = numerator/denominator1 ## bare(!)

        ## smoothness prior
        tk = np.log(pk)
        Amemory = None
        var_ = var*1.1 # temporally increasing the variance
        breakinfo = False
        while(var_>=var): # slowly lowering the variance
            absdelta = 1
            while(absdelta>1E-3): # solving with fixed variance
                ## solution of A delta = b1 - b2
                Ainverse,denominator2,Amemory = _calc_inverse(tk,var_,kindex,rho,denominator1,Amemory)
                delta = np.dot(Ainverse,numerator/pk-denominator2,out=None)
                if(np.abs(delta).max()>absdelta): # increasing variance when speeding up
                    var_ *= 1.1
                absdelta = np.abs(delta).max()
                tk += min(1,0.1/absdelta)*delta # adaptive step width
                pk *= np.exp(min(1,0.1/absdelta)*delta) # adaptive step width
            var_ /= 1.1 # lowering the variance when converged
            if(var_<var):
                if(breakinfo): # making sure there's one iteration with the correct variance
                    break
                var_ = var
                breakinfo = True

        ## weight if ...
        if(not domain.discrete)and(not bare):
            pk = weight_power(domain,pk,power=1,pindex=pindex,pundex=pundex) ## non-bare(!)

    else:
        pk = numerator/denominator1 ## non-bare(!)
        ## weight if ...
        if(not domain.discrete)and(not bare):
            pk = weight_power(domain,pk,power=1,pindex=pindex,pundex=pundex) ## non-bare(!)

    return pk

##=============================================================================

