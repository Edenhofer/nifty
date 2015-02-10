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
    ..     /__/ /__/ /__/ /__/    \___/  \___   /  tools
    ..                                  /______/

    This module extends NIFTY with a nifty set of tools including further
    operators, namely the :py:class:`invertible_operator` and the
    :py:class:`propagator_operator`, and minimization schemes, namely
    :py:class:`steepest_descent` and :py:class:`conjugate_gradient`. Those
    tools are supposed to support the user in solving information field
    theoretical problems (almost) without numerical pain.

"""
from __future__ import division
#from nifty_core import *
import numpy as np
from nifty_core import notification,about,                                   \
                       space,                                                \
                       field,                                                \
                       operator,diagonal_operator


##-----------------------------------------------------------------------------

class invertible_operator(operator):
    """
        ..       __                                       __     __   __        __
        ..     /__/                                     /  /_  /__/ /  /      /  /
        ..     __   __ ___  __   __   _______   _____  /   _/  __  /  /___   /  /   _______
        ..   /  / /   _   ||  |/  / /   __  / /   __/ /  /   /  / /   _   | /  /  /   __  /
        ..  /  / /  / /  / |     / /  /____/ /  /    /  /_  /  / /  /_/  / /  /_ /  /____/
        .. /__/ /__/ /__/  |____/  \______/ /__/     \___/ /__/  \______/  \___/ \______/  operator class

        NIFTY subclass for invertible, self-adjoint (linear) operators

        The invertible operator class is an abstract class for self-adjoint or
        symmetric (linear) operators from which other more specific operator
        subclassescan be derived. Such operators inherit an automated inversion
        routine, namely conjugate gradient.

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.
        uni : bool, *optional*
            Indicates whether the operator is unitary or not.
            (default: False)
        imp : bool, *optional*
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not (default: False).
        para : {single object, tuple/list of objects}, *optional*
            This is a freeform tuple/list of parameters that derivatives of
            the operator class can use (default: None).

        See Also
        --------
        operator

        Notes
        -----
        This class is not meant to be instantiated. Operator classes derived
        from this one only need a `_multiply` or `_inverse_multiply` instance
        method to perform the other. However, one of them needs to be defined.

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        sym : bool
            Indicates whether the operator is self-adjoint or not.
        uni : bool
            Indicates whether the operator is unitary or not.
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not.
        target : space
            The space wherein the operator output lives.
        para : {single object, list of objects}
            This is a freeform tuple/list of parameters that derivatives of
            the operator class can use. Not used in the base operators.

    """
    def __init__(self,domain,uni=False,imp=False,para=None):
        """
            Sets the standard operator properties.

            Parameters
            ----------
            domain : space
                The space wherein valid arguments live.
            uni : bool, *optional*
                Indicates whether the operator is unitary or not.
                (default: False)
            imp : bool, *optional*
                Indicates whether the incorporation of volume weights in
                multiplications is already implemented in the `multiply`
                instance methods or not (default: False).
            para : {single object, tuple/list of objects}, *optional*
                This is a freeform tuple/list of parameters that derivatives of
                the operator class can use (default: None).

        """
        if(not isinstance(domain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = domain
        self.sym = True
        self.uni = bool(uni)

        if(self.domain.discrete):
            self.imp = True
        else:
            self.imp = bool(imp)

        self.target = self.domain

        if(para is not None):
            self.para = para

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _multiply(self,x,force=False,W=None,spam=None,reset=None,note=False,x0=None,tol=1E-4,clevel=1,limii=None,**kwargs):
        """
            Applies the invertible operator to a given field by invoking a
            conjugate gradient.

            Parameters
            ----------
            x : field
                Valid input field.
            force : bool
                Indicates wheter to return a field instead of ``None`` in case
                the conjugate gradient fails.

            Returns
            -------
            OIIx : field
                Mapped field with suitable domain.

            See Also
            --------
            conjugate_gradient

            Other Parameters
            ----------------
            W : {operator, function}, *optional*
                Operator `W` that is a preconditioner on `A` and is applicable to a
                field (default: None).
            spam : function, *optional*
                Callback function which is given the current `x` and iteration
                counter each iteration (default: None).
            reset : integer, *optional*
                Number of iterations after which to restart; i.e., forget previous
                conjugated directions (default: sqrt(b.dim())).
            note : bool, *optional*
                Indicates whether notes are printed or not (default: False).
            x0 : field, *optional*
                Starting guess for the minimization.
            tol : scalar, *optional*
                Tolerance specifying convergence; measured by current relative
                residual (default: 1E-4).
            clevel : integer, *optional*
                Number of times the tolerance should be undershot before
                exiting (default: 1).
            limii : integer, *optional*
                Maximum number of iterations performed (default: 10 * b.dim()).

        """
        x_,convergence = conjugate_gradient(self.inverse_times,x,W=W,spam=spam,reset=reset,note=note)(x0=x0,tol=tol,clevel=clevel,limii=limii)
        ## check convergence
        if(not convergence):
            if(not force)or(x_ is None):
                return None
            about.warnings.cprint("WARNING: conjugate gradient failed.")
        ## weight if ...
        if(not self.imp): ## continiuos domain/target
            x_.weight(power=-1,overwrite=True)
        return x_

    def _inverse_multiply(self,x,force=False,W=None,spam=None,reset=None,note=False,x0=None,tol=1E-4,clevel=1,limii=None,**kwargs):
        """
            Applies the inverse of the invertible operator to a given field by
            invoking a conjugate gradient.

            Parameters
            ----------
            x : field
                Valid input field.
            force : bool
                Indicates wheter to return a field instead of ``None`` in case
                the conjugate gradient fails.

            Returns
            -------
            OIx : field
                Mapped field with suitable domain.

            See Also
            --------
            conjugate_gradient

            Other Parameters
            ----------------
            W : {operator, function}, *optional*
                Operator `W` that is a preconditioner on `A` and is applicable to a
                field (default: None).
            spam : function, *optional*
                Callback function which is given the current `x` and iteration
                counter each iteration (default: None).
            reset : integer, *optional*
                Number of iterations after which to restart; i.e., forget previous
                conjugated directions (default: sqrt(b.dim())).
            note : bool, *optional*
                Indicates whether notes are printed or not (default: False).
            x0 : field, *optional*
                Starting guess for the minimization.
            tol : scalar, *optional*
                Tolerance specifying convergence; measured by current relative
                residual (default: 1E-4).
            clevel : integer, *optional*
                Number of times the tolerance should be undershot before
                exiting (default: 1).
            limii : integer, *optional*
                Maximum number of iterations performed (default: 10 * b.dim()).

        """
        x_,convergence = conjugate_gradient(self.times,x,W=W,spam=spam,reset=reset,note=note)(x0=x0,tol=tol,clevel=clevel,limii=limii)
        ## check convergence
        if(not convergence):
            if(not force)or(x_ is None):
                return None
            about.warnings.cprint("WARNING: conjugate gradient failed.")
        ## weight if ...
        if(not self.imp): ## continiuos domain/target
            x_.weight(power=1,overwrite=True)
        return x_

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_tools.invertible_operator>"

##-----------------------------------------------------------------------------

##-----------------------------------------------------------------------------

class propagator_operator(operator):
    """
        ..                                                                            __
        ..                                                                          /  /_
        ..      _______   _____   ______    ______    ____ __   ____ __   ____ __  /   _/  ______    _____
        ..    /   _   / /   __/ /   _   | /   _   | /   _   / /   _   / /   _   / /  /   /   _   | /   __/
        ..   /  /_/  / /  /    /  /_/  / /  /_/  / /  /_/  / /  /_/  / /  /_/  / /  /_  /  /_/  / /  /
        ..  /   ____/ /__/     \______/ /   ____/  \______|  \___   /  \______|  \___/  \______/ /__/     operator class
        .. /__/                        /__/                 /______/

        NIFTY subclass for propagator operators (of a certain family)

        The propagator operators :math:`D` implemented here have an inverse
        formulation like :math:`(S^{-1} + M)`, :math:`(S^{-1} + N^{-1})`, or
        :math:`(S^{-1} + R^\dagger N^{-1} R)` as appearing in Wiener filter
        theory.

        Parameters
        ----------
        S : operator
            Covariance of the signal prior.
        M : operator
            Likelihood contribution.
        R : operator
            Response operator translating signal to (noiseless) data.
        N : operator
            Covariance of the noise prior or the likelihood, respectively.

        See Also
        --------
        conjugate_gradient

        Notes
        -----
        The propagator will puzzle the operators `S` and `M` or `R`, `N` or
        only `N` together in the predefined from, a domain is set
        automatically. The application of the inverse is done by invoking a
        conjugate gradient.
        Note that changes to `S`, `M`, `R` or `N` auto-update the propagator.

        Examples
        --------
        >>> f = field(rg_space(4), val=[2, 4, 6, 8])
        >>> S = power_operator(f.target, spec=1)
        >>> N = diagonal_operator(f.domain, diag=1)
        >>> D = propagator_operator(S=S, N=N) # D^{-1} = S^{-1} + N^{-1}
        >>> D(f).val
        array([ 1.,  2.,  3.,  4.])

        Attributes
        ----------
        domain : space
            A space wherein valid arguments live.
        codomain : space
            An alternative space wherein valid arguments live; commonly the
            codomain of the `domain` attribute.
        sym : bool
            Indicates that the operator is self-adjoint.
        uni : bool
            Indicates that the operator is not unitary.
        imp : bool
            Indicates that volume weights are implemented in the `multiply`
            instance methods.
        target : space
            The space wherein the operator output lives.
        _A1 : {operator, function}
            Application of :math:`S^{-1}` to a field.
        _A2 : {operator, function}
            Application of all operations not included in `A1` to a field.
        RN : {2-tuple of operators}, *optional*
            Contains `R` and `N` if given.

    """
    def __init__(self,S=None,M=None,R=None,N=None):
        """
            Sets the standard operator properties and `codomain`, `_A1`, `_A2`,
            and `RN` if required.

            Parameters
            ----------
            S : operator
                Covariance of the signal prior.
            M : operator
                Likelihood contribution.
            R : operator
                Response operator translating signal to (noiseless) data.
            N : operator
                Covariance of the noise prior or the likelihood, respectively.

        """
        ## check signal prior covariance
        if(S is None):
            raise Exception(about._errors.cstring("ERROR: insufficient input."))
        elif(not isinstance(S,operator)):
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
        space1 = S.domain

        ## check likelihood (pseudo) covariance
        if(M is None):
            if(N is None):
                raise Exception(about._errors.cstring("ERROR: insufficient input."))
            elif(not isinstance(N,operator)):
                raise ValueError(about._errors.cstring("ERROR: invalid input."))
            if(R is None):
                space2 = N.domain
            elif(not isinstance(R,operator)):
                raise ValueError(about._errors.cstring("ERROR: invalid input."))
            else:
                space2 = R.domain
        elif(not isinstance(M,operator)):
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
        else:
            space2 = M.domain

        ## set spaces
        self.domain = space2
        if(self.domain.check_codomain(space1)):
            self.codomain = space1
        else:
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
        self.target = self.domain

        ## define A1 == S_inverse
        if(isinstance(S,diagonal_operator)):
            self._A1 = S._inverse_multiply ## S.imp == True
        else:
            self._A1 = S.inverse_times

        ## define A2 == M == R_adjoint N_inverse R == N_inverse
        if(M is None):
            if(R is not None):
                self.RN = (R,N)
                if(isinstance(N,diagonal_operator)):
                    self._A2 = self._standard_M_times_1
                else:
                    self._A2 = self._standard_M_times_2
            elif(isinstance(N,diagonal_operator)):
                self._A2 = N._inverse_multiply ## N.imp == True
            else:
                self._A2 = N.inverse_times
        elif(isinstance(M,diagonal_operator)):
            self._A2 = M._multiply ## M.imp == True
        else:
            self._A2 = M.times

        self.sym = True
        self.uni = False
        self.imp = True

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _standard_M_times_1(self,x,**kwargs): ## applies > R_adjoint N_inverse R assuming N is diagonal
        return self.RN[0].adjoint_times(self.RN[1]._inverse_multiply(self.RN[0].times(x))) ## N.imp = True

    def _standard_M_times_2(self,x,**kwargs): ## applies > R_adjoint N_inverse R
        return self.RN[0].adjoint_times(self.RN[1].inverse_times(self.RN[0].times(x)))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _inverse_multiply_1(self,x,**kwargs): ## > applies A1 + A2 in self.codomain
        return self._A1(x,pseudo=True)+self._A2(x.transform(self.domain)).transform(self.codomain)

    def _inverse_multiply_2(self,x,**kwargs): ## > applies A1 + A2 in self.domain
        return self._A1(x.transform(self.codomain),pseudo=True).transform(self.domain)+self._A2(x)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _briefing(self,x): ## > prepares x for `multiply`
        ## inspect x
        if(not isinstance(x,field)):
            return field(self.domain,val=x,target=None),False
        ## check x.domain
        elif(x.domain==self.domain):
            return x,False
        elif(x.domain==self.codomain):
            return x,True
        ## transform
        else:
            return x.transform(target=self.codomain,overwrite=False),True

    def _debriefing(self,x,x_,in_codomain): ## > evaluates x and x_ after `multiply`
        if(x_ is None):
            return None
        ## inspect x
        elif(isinstance(x,field)):
            ## repair ...
            if(in_codomain)and(x.domain!=self.codomain):
                    x_ = x_.transform(target=x.domain,overwrite=False) ## ... domain
            if(x_.target!=x.target):
                x_.set_target(newtarget=x.target) ## ... codomain
        return x_

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def times(self,x,force=False,W=None,spam=None,reset=None,note=False,x0=None,tol=1E-4,clevel=1,limii=None,**kwargs):
        """
            Applies the propagator to a given object by invoking a
            conjugate gradient.

            Parameters
            ----------
            x : {scalar, list, array, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the domain of the operator.
            force : bool
                Indicates wheter to return a field instead of ``None`` in case
                the conjugate gradient fails.

            Returns
            -------
            Dx : field
                Mapped field with suitable domain.

            See Also
            --------
            conjugate_gradient

            Other Parameters
            ----------------
            W : {operator, function}, *optional*
                Operator `W` that is a preconditioner on `A` and is applicable to a
                field (default: None).
            spam : function, *optional*
                Callback function which is given the current `x` and iteration
                counter each iteration (default: None).
            reset : integer, *optional*
                Number of iterations after which to restart; i.e., forget previous
                conjugated directions (default: sqrt(b.dim())).
            note : bool, *optional*
                Indicates whether notes are printed or not (default: False).
            x0 : field, *optional*
                Starting guess for the minimization.
            tol : scalar, *optional*
                Tolerance specifying convergence; measured by current relative
                residual (default: 1E-4).
            clevel : integer, *optional*
                Number of times the tolerance should be undershot before
                exiting (default: 1).
            limii : integer, *optional*
                Maximum number of iterations performed (default: 10 * b.dim()).

        """
        ## prepare
        x_,in_codomain = self._briefing(x)
        ## apply operator
        if(in_codomain):
            A = self._inverse_multiply_1
        else:
            A = self._inverse_multiply_2
        x_,convergence = conjugate_gradient(A,x_,W=W,spam=spam,reset=reset,note=note)(x0=x0,tol=tol,clevel=clevel,limii=limii)
        ## evaluate
        if(not convergence):
            if(not force):
                return None
            about.warnings.cprint("WARNING: conjugate gradient failed.")
        return self._debriefing(x,x_,in_codomain)

    def inverse_times(self,x,**kwargs):
        """
            Applies the inverse propagator to a given object.

            Parameters
            ----------
            x : {scalar, list, array, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the domain of the operator.

            Returns
            -------
            DIx : field
                Mapped field with suitable domain.

        """
        ## prepare
        x_,in_codomain = self._briefing(x)
        ## apply operator
        if(in_codomain):
            x_ = self._inverse_multiply_1(x_)
        else:
            x_ = self._inverse_multiply_2(x_)
        ## evaluate
        return self._debriefing(x,x_,in_codomain)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_tools.propagator_operator>"

##-----------------------------------------------------------------------------

##=============================================================================

class conjugate_gradient(object):
    """
        ..      _______       ____ __
        ..    /  _____/     /   _   /
        ..   /  /____  __  /  /_/  / __
        ..   \______//__/  \____  //__/  class
        ..                /______/

        NIFTY tool class for conjugate gradient

        This tool minimizes :math:`A x = b` with respect to `x` given `A` and
        `b` using a conjugate gradient; i.e., a step-by-step minimization
        relying on conjugated gradient directions. Further, `A` is assumed to
        be a positive definite and self-adjoint operator. The use of a
        preconditioner `W` that is roughly the inverse of `A` is optional.
        For details on the methodology refer to [#]_, for details on usage and
        output, see the notes below.

        Parameters
        ----------
        A : {operator, function}
            Operator `A` applicable to a field.
        b : field
            Resulting field of the operation `A(x)`.
        W : {operator, function}, *optional*
            Operator `W` that is a preconditioner on `A` and is applicable to a
            field (default: None).
        spam : function, *optional*
            Callback function which is given the current `x` and iteration
            counter each iteration (default: None).
        reset : integer, *optional*
            Number of iterations after which to restart; i.e., forget previous
            conjugated directions (default: sqrt(b.dim())).
        note : bool, *optional*
            Indicates whether notes are printed or not (default: False).

        See Also
        --------
        scipy.sparse.linalg.cg

        Notes
        -----
        After initialization by `__init__`, the minimizer is started by calling
        it using `__call__`, which takes additional parameters. Notifications,
        if enabled, will state the iteration number, current step widths
        `alpha` and `beta`, the current relative residual `delta` that is
        compared to the tolerance, and the convergence level if changed.
        The minimizer will exit in three states: DEAD if alpha becomes
        infinite, QUIT if the maximum number of iterations is reached, or DONE
        if convergence is achieved. Returned will be the latest `x` and the
        latest convergence level, which can evaluate ``True`` for the exit
        states QUIT and DONE.

        References
        ----------
        .. [#] J. R. Shewchuk, 1994, `"An Introduction to the Conjugate
            Gradient Method Without the Agonizing Pain"
            <http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf>`_

        Examples
        --------
        >>> b = field(point_space(2), val=[1, 9])
        >>> A = diagonal_operator(b.domain, diag=[4, 3])
        >>> x,convergence = conjugate_gradient(A, b, note=True)(tol=1E-4, clevel=3)
        iteration : 00000001   alpha = 3.3E-01   beta = 1.3E-03   delta = 3.6E-02
        iteration : 00000002   alpha = 2.5E-01   beta = 7.6E-04   delta = 1.0E-03
        iteration : 00000003   alpha = 3.3E-01   beta = 2.5E-04   delta = 1.6E-05   convergence level : 1
        iteration : 00000004   alpha = 2.5E-01   beta = 1.8E-06   delta = 2.1E-08   convergence level : 2
        iteration : 00000005   alpha = 2.5E-01   beta = 2.2E-03   delta = 1.0E-09   convergence level : 3
        ... done.
        >>> bool(convergence)
        True
        >>> x.val # yields 1/4 and 9/3
        array([ 0.25,  3.  ])

        Attributes
        ----------
        A : {operator, function}
            Operator `A` applicable to a field.
        x : field
            Current field.
        b : field
            Resulting field of the operation `A(x)`.
        W : {operator, function}
            Operator `W` that is a preconditioner on `A` and is applicable to a
            field; can be ``None``.
        spam : function
            Callback function which is given the current `x` and iteration
            counter each iteration; can be ``None``.
        reset : integer
            Number of iterations after which to restart; i.e., forget previous
            conjugated directions (default: sqrt(b.dim())).
        note : notification
            Notification instance.

    """
    def __init__(self,A,b,W=None,spam=None,reset=None,note=False):
        """
            Initializes the conjugate_gradient and sets the attributes (except
            for `x`).

            Parameters
            ----------
            A : {operator, function}
                Operator `A` applicable to a field.
            b : field
                Resulting field of the operation `A(x)`.
            W : {operator, function}, *optional*
                Operator `W` that is a preconditioner on `A` and is applicable to a
                field (default: None).
            spam : function, *optional*
                Callback function which is given the current `x` and iteration
                counter each iteration (default: None).
            reset : integer, *optional*
                Number of iterations after which to restart; i.e., forget previous
                conjugated directions (default: sqrt(b.dim())).
            note : bool, *optional*
                Indicates whether notes are printed or not (default: False).

        """
        if(hasattr(A,"__call__")):
            self.A = A ## applies A
        else:
            raise AttributeError(about._errors.cstring("ERROR: invalid input."))
        self.b = b

        if(W is None)or(hasattr(W,"__call__")):
            self.W = W ## applies W ~ A_inverse
        else:
            raise AttributeError(about._errors.cstring("ERROR: invalid input."))

        self.spam = spam ## serves as callback given x and iteration number
        if(reset is None): ## 2 < reset ~ sqrt(dim)
            self.reset = max(2,int(np.sqrt(b.domain.dim(split=False))))
        else:
            self.reset = max(2,int(reset))
        self.note = notification(default=bool(note))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __call__(self,x0=None,**kwargs): ## > runs cg with/without preconditioner
        """
            Runs the conjugate gradient minimization.

            Parameters
            ----------
            x0 : field, *optional*
                Starting guess for the minimization.
            tol : scalar, *optional*
                Tolerance specifying convergence; measured by current relative
                residual (default: 1E-4).
            clevel : integer, *optional*
                Number of times the tolerance should be undershot before
                exiting (default: 1).
            limii : integer, *optional*
                Maximum number of iterations performed (default: 10 * b.dim()).

            Returns
            -------
            x : field
                Latest `x` of the minimization.
            convergence : integer
                Latest convergence level indicating whether the minimization
                has converged or not.

        """
        self.x = field(self.b.domain,val=x0,target=self.b.target)

        if(self.W is None):
            return self._calc_without(**kwargs)
        else:
            return self._calc_with(**kwargs)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _calc_without(self,tol=1E-4,clevel=1,limii=None): ## > runs cg without preconditioner
        clevel = int(clevel)
        if(limii is None):
            limii = 10*self.b.domain.dim(split=False)
        else:
            limii = int(limii)

        r = self.b-self.A(self.x)
        d = field(self.b.domain,val=np.copy(r.val),target=self.b.target)
        gamma = r.dot(d)
        if(gamma==0):
            return self.x,clevel+1
        delta_ = np.absolute(gamma)**(-0.5)

        convergence = 0
        ii = 1
        while(True):
            q = self.A(d)
            alpha = gamma/d.dot(q) ## positive definite
            if(not np.isfinite(alpha)):
                self.note.cprint("\niteration : %08u   alpha = NAN\n... dead."%ii)
                return self.x,0
            self.x += alpha*d
            if(np.signbit(np.real(alpha))):
                about.warnings.cprint("WARNING: positive definiteness of A violated.")
                r = self.b-self.A(self.x)
            elif(ii%self.reset==0):
                r = self.b-self.A(self.x)
            else:
                r -= alpha*q
            gamma_ = gamma
            gamma = r.dot(r)
            beta = max(0,gamma/gamma_) ## positive definite
            d = r+beta*d

            delta = delta_*np.absolute(gamma)**0.5
            self.note.cflush("\niteration : %08u   alpha = %3.1E   beta = %3.1E   delta = %3.1E"%(ii,np.real(alpha),np.real(beta),np.real(delta)))
            if(gamma==0):
                convergence = clevel+1
                self.note.cprint("   convergence level : INF\n... done.")
                break
            elif(np.absolute(delta)<tol):
                convergence += 1
                self.note.cflush("   convergence level : %u"%convergence)
                if(convergence==clevel):
                    self.note.cprint("\n... done.")
                    break
            else:
                convergence = max(0,convergence-1)
            if(ii==limii):
                self.note.cprint("\n... quit.")
                break

            if(self.spam is not None):
                self.spam(self.x,ii)

            ii += 1

        if(self.spam is not None):
            self.spam(self.x,ii)

        return self.x,convergence

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _calc_with(self,tol=1E-4,clevel=1,limii=None): ## > runs cg with preconditioner

        clevel = int(clevel)
        if(limii is None):
            limii = 10*self.b.domain.dim(split=False)
        else:
            limii = int(limii)

        r = self.b-self.A(self.x)
        d = self.W(r)
        gamma = r.dot(d)
        if(gamma==0):
            return self.x,clevel+1
        delta_ = np.absolute(gamma)**(-0.5)

        convergence = 0
        ii = 1
        while(True):
            q = self.A(d)
            alpha = gamma/d.dot(q) ## positive definite
            if(not np.isfinite(alpha)):
                self.note.cprint("\niteration : %08u   alpha = NAN\n... dead."%ii)
                return self.x,0
            self.x += alpha*d ## update
            if(np.signbit(np.real(alpha))):
                about.warnings.cprint("WARNING: positive definiteness of A violated.")
                r = self.b-self.A(self.x)
            elif(ii%self.reset==0):
                r = self.b-self.A(self.x)
            else:
                r -= alpha*q
            s = self.W(r)
            gamma_ = gamma
            gamma = r.dot(s)
            if(np.signbit(np.real(gamma))):
                about.warnings.cprint("WARNING: positive definiteness of W violated.")
            beta = max(0,gamma/gamma_) ## positive definite
            d = s+beta*d ## conjugated gradient

            delta = delta_*np.absolute(gamma)**0.5
            self.note.cflush("\niteration : %08u   alpha = %3.1E   beta = %3.1E   delta = %3.1E"%(ii,np.real(alpha),np.real(beta),np.real(delta)))
            if(gamma==0):
                convergence = clevel+1
                self.note.cprint("   convergence level : INF\n... done.")
                break
            elif(np.absolute(delta)<tol):
                convergence += 1
                self.note.cflush("   convergence level : %u"%convergence)
                if(convergence==clevel):
                    self.note.cprint("\n... done.")
                    break
            else:
                convergence = max(0,convergence-1)
            if(ii==limii):
                self.note.cprint("\n... quit.")
                break

            if(self.spam is not None):
                self.spam(self.x,ii)

            ii += 1

        if(self.spam is not None):
            self.spam(self.x,ii)

        return self.x,convergence

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_tools.conjugate_gradient>"

##=============================================================================





##=============================================================================

class steepest_descent(object):
    """
        ..                          __
        ..                        /  /
        ..      _______      ____/  /
        ..    /  _____/    /   _   /
        ..   /_____  / __ /  /_/  / __
        ..  /_______//__/ \______|/__/  class

        NIFTY tool class for steepest descent minimization

        This tool minimizes a scalar energy-function by steepest descent using
        the functions gradient. Steps and step widths are choosen according to
        the Wolfe conditions [#]_. For details on usage and output, see the
        notes below.

        Parameters
        ----------
        eggs : function
            Given the current `x` it returns the tuple of energy and gradient.
        spam : function, *optional*
            Callback function which is given the current `x` and iteration
            counter each iteration (default: None).
        a : {4-tuple}, *optional*
            Numbers obeying 0 < a1 ~ a2 < 1 ~ a3 < a4 that modify the step
            widths (default: (0.2,0.5,1,2)).
        c : {2-tuple}, *optional*
            Numbers obeying 0 < c1 < c2 < 1 that specify the Wolfe-conditions
            (default: (1E-4,0.9)).
        note : bool, *optional*
            Indicates whether notes are printed or not (default: False).

        See Also
        --------
        scipy.optimize.fmin_cg, scipy.optimize.fmin_ncg,
        scipy.optimize.fmin_l_bfgs_b

        Notes
        -----
        After initialization by `__init__`, the minimizer is started by calling
        it using `__call__`, which takes additional parameters. Notifications,
        if enabled, will state the iteration number, current step width `alpha`,
        current maximal change `delta` that is compared to the tolerance, and
        the convergence level if changed. The minimizer will exit in three
        states: DEAD if no step width above 1E-13 is accepted, QUIT if the
        maximum number of iterations is reached, or DONE if convergence is
        achieved. Returned will be the latest `x` and the latest convergence
        level, which can evaluate ``True`` for all exit states.

        References
        ----------
        .. [#] J. Nocedal and S. J. Wright, Springer 2006, "Numerical
            Optimization", ISBN: 978-0-387-30303-1 (print) / 978-0-387-40065-5
            `(online) <http://link.springer.com/book/10.1007/978-0-387-40065-5/page/1>`_

        Examples
        --------
        >>> def egg(x):
        ...     E = 0.5*x.dot(x) # energy E(x) -- a two-dimensional parabola
        ...     g = x # gradient
        ...     return E,g
        >>> x = field(point_space(2), val=[1, 3])
        >>> x,convergence = steepest_descent(egg, note=True)(x0=x, tol=1E-4, clevel=3)
        iteration : 00000001   alpha = 1.0E+00   delta = 6.5E-01
        iteration : 00000002   alpha = 2.0E+00   delta = 1.4E-01
        iteration : 00000003   alpha = 1.6E-01   delta = 2.1E-03
        iteration : 00000004   alpha = 2.6E-03   delta = 3.0E-04
        iteration : 00000005   alpha = 2.0E-04   delta = 5.3E-05   convergence level : 1
        iteration : 00000006   alpha = 8.2E-05   delta = 4.4E-06   convergence level : 2
        iteration : 00000007   alpha = 6.6E-06   delta = 3.1E-06   convergence level : 3
        ... done.
        >>> bool(convergence)
        True
        >>> x.val # approximately zero
        array([ -6.87299426e-07  -2.06189828e-06])

        Attributes
        ----------
        x : field
            Current field.
        eggs : function
            Given the current `x` it returns the tuple of energy and gradient.
        spam : function
            Callback function which is given the current `x` and iteration
            counter each iteration; can be ``None``.
        a : {4-tuple}
            Numbers obeying 0 < a1 ~ a2 < 1 ~ a3 < a4 that modify the step
            widths (default: (0.2,0.5,1,2)).
        c : {2-tuple}
            Numbers obeying 0 < c1 < c2 < 1 that specify the Wolfe-conditions
            (default: (1E-4,0.9)).
        note : notification
            Notification instance.

    """
    def __init__(self,eggs,spam=None,a=(0.2,0.5,1,2),c=(1E-4,0.9),note=False):
        """
            Initializes the steepest_descent and sets the attributes (except
            for `x`).

            Parameters
            ----------
            eggs : function
                Given the current `x` it returns the tuple of energy and gradient.
            spam : function, *optional*
                Callback function which is given the current `x` and iteration
                counter each iteration (default: None).
            a : {4-tuple}, *optional*
                Numbers obeying 0 < a1 ~ a2 < 1 ~ a3 < a4 that modify the step
                widths (default: (0.2,0.5,1,2)).
            c : {2-tuple}, *optional*
                Numbers obeying 0 < c1 < c2 < 1 that specify the Wolfe-conditions
                (default: (1E-4,0.9)).
            note : bool, *optional*
                Indicates whether notes are printed or not (default: False).

        """
        self.eggs = eggs ## returns energy and gradient

        self.spam = spam ## serves as callback given x and iteration number
        self.a = a ## 0 < a1 ~ a2 < 1 ~ a3 < a4
        self.c = c ## 0 < c1 < c2 < 1
        self.note = notification(default=bool(note))

        self._alpha = None ## last alpha

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __call__(self,x0,alpha=1,tol=1E-4,clevel=8,limii=100000):
        """
            Runs the steepest descent minimization.

            Parameters
            ----------
            x0 : field
                Starting guess for the minimization.
            alpha : scalar, *optional*
                Starting step width to be multiplied with normalized gradient
                (default: 1).
            tol : scalar, *optional*
                Tolerance specifying convergence; measured by maximal change in
                `x` (default: 1E-4).
            clevel : integer, *optional*
                Number of times the tolerance should be undershot before
                exiting (default: 8).
            limii : integer, *optional*
                Maximum number of iterations performed (default: 100,000).

            Returns
            -------
            x : field
                Latest `x` of the minimization.
            convergence : integer
                Latest convergence level indicating whether the minimization
                has converged or not.

        """
        if(not isinstance(x0,field)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.x = x0

        ## check for exsisting alpha
        if(alpha is None):
            if(self._alpha is not None):
                alpha = self._alpha
            else:
                alpha = 1

        clevel = max(1,int(clevel))
        limii = int(limii)

        E,g = self.eggs(self.x) ## energy and gradient
        norm = g.norm() ## gradient norm
        if(norm==0):
            self.note.cprint("\niteration : 00000000   alpha = 0.0E+00   delta = 0.0E+00\n... done.")
            return self.x,clevel+2

        convergence = 0
        ii = 1
        while(True):
            x_,E,g,alpha,a = self._get_alpha(E,g,norm,alpha) ## "news",alpha,a

            if(alpha is None):
                self.note.cprint("\niteration : %08u   alpha < 1.0E-13\n... dead."%ii)
                break
            else:
                delta = np.absolute(g.val).max()*(alpha/norm)
                self.note.cflush("\niteration : %08u   alpha = %3.1E   delta = %3.1E"%(ii,alpha,delta))
                ## update
                self.x = x_
                alpha *= a

            norm = g.norm() ## gradient norm
            if(delta==0):
                convergence = clevel+2
                self.note.cprint("   convergence level : %u\n... done."%convergence)
                break
            elif(delta<tol):
                convergence += 1
                self.note.cflush("   convergence level : %u"%convergence)
                if(convergence==clevel):
                    convergence += int(ii==clevel)
                    self.note.cprint("\n... done.")
                    break
            else:
                convergence = max(0,convergence-1)
            if(ii==limii):
                self.note.cprint("\n... quit.")
                break

            if(self.spam is not None):
                self.spam(self.x,ii)

            ii += 1

        if(self.spam is not None):
            self.spam(self.x,ii)

        ## memorise last alpha
        if(alpha is not None):
            self._alpha = alpha/a ## undo update

        return self.x,convergence

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _get_alpha(self,E,g,norm,alpha): ## > determines the new alpha
        while(True):
            ## Wolfe conditions
            wolfe,x_,E_,g_,a = self._check_wolfe(E,g,norm,alpha)
#            wolfe,x_,E_,g_,a = self._check_strong_wolfe(E,g,norm,alpha)
            if(wolfe):
                return x_,E_,g_,alpha,a
            else:
                alpha *= a
                if(alpha<1E-13):
                    return None,None,None,None,None

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _check_wolfe(self,E,g,norm,alpha): ## > checks the Wolfe conditions
        x_ = self._get_x(g,norm,alpha)
        pg = norm
        E_,g_ = self.eggs(x_)
        if(E_>E+self.c[0]*alpha*pg):
            if(E_<E):
                return True,x_,E_,g_,self.a[1]
            return False,None,None,None,self.a[0]
        pg_ = g.dot(g_)/norm
        if(pg_<self.c[1]*pg):
            return True,x_,E_,g_,self.a[3]
        return True,x_,E_,g_,self.a[2]

#    def _check_strong_wolfe(self,E,g,norm,alpha): ## > checks the strong Wolfe conditions
#        x_ = self._get_x(g,norm,alpha)
#        pg = norm
#        E_,g_ = self.eggs(x_)
#        if(E_>E+self.c[0]*alpha*pg):
#            if(E_<E):
#                return True,x_,E_,g_,self.a[1]
#            return False,None,None,None,self.a[0]
#        apg_ = np.absolute(g.dot(g_))/norm
#        if(apg_>self.c[1]*np.absolute(pg)):
#            return True,x_,E_,g_,self.a[3]
#        return True,x_,E_,g_,self.a[2]

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _get_x(self,g,norm,alpha): ## > updates x
        return self.x-g*(alpha/norm)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_tools.steepest_descent>"

##=============================================================================

