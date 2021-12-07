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
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras, Philipp Frank

import os
import pickle
import re
import time

from .. import utilities
from ..field import Field
from ..multi_domain import MultiDomain
from ..multi_field import MultiField
from ..operators.operator import Operator


class SampleListBase:
    """Base class for storing lists of fields representing samples.

    This class suits as a base class for storing lists of in most cases
    posterior samples. It is intended to be used to hold the minimization state
    of an inference run and comes with a variety of convenience functions like
    computing the mean or standard deviation of the output of a given operator
    over the sample list.

    Parameters
    ----------
    comm : MPI communicator or None
        If not `None`, :class:`SampleListBase` can gather samples across multiple
        MPI tasks. If `None`, :class:`SampleListBase` is not a distributed object.
    domain : Domainoid (can be DomainTuple, MultiDomain, dict, Domain or list of Domains)
        The domain on which the samples are defined.

    Note
    ----
    A class inheriting from :class:`SampleListBase` needs to call the constructor of
    `SampleListBase` and needs to implement :attr:`n_local_samples` and :attr:`local_item()`.
    """
    def __init__(self, comm, domain):
        from ..sugar import makeDomain
        self._comm = comm
        self._domain = makeDomain(domain)
        utilities.check_MPI_equality(self._domain, comm)

    @property
    def n_local_samples(self):
        """int: Number of local samples."""
        raise NotImplementedError

    def n_samples(self):
        """Return number of samples across all MPI tasks."""
        return utilities.allreduce_sum([self.n_local_samples], self.comm)

    def local_item(self, i):
        """Return ith local sample."""
        raise NotImplementedError

    def local_iterator(self):
        """Return an iterator over all local samples."""
        for i in range(self.n_local_samples):
            yield self.local_item(i)

    @property
    def comm(self):
        """MPI communicator or None: The communicator used for the SampleListBase."""
        return self._comm

    @property
    def MPI_master(self):
        return utilities.get_MPI_params_from_comm(self.comm)[2]

    @property
    def domain(self):
        """DomainTuple or MultiDomain: the domain on which the samples are defined."""
        return self._domain

    @staticmethod
    def local_indices(n_samples, comm=None):
        """Return range of global sample indices for local task.

        This method calls `utilities.shareRange`

        Parameters
        ----------
        n_samples : int
            Number of work items to be distributed.
        comm : MPI communicator or None
            The communicator used for the distribution.

        Returns
        -------
        range
            Range of relevant indices for the local task.
        """
        ntask, rank, _ = utilities.get_MPI_params_from_comm(comm)
        return range(*utilities.shareRange(n_samples, ntask, rank))

    def save_to_hdf5(self, file_name, op=None, samples=False, mean=False, std=False,
                     overwrite=False):
        """Write sample list to HDF5 file.

        This function writes sample lists to HDF5 files that contain two
        groups: `samples` and `stats`. `samples` contain the sublabels `0`,
        `1`, ... that number the labels and `stats` contains the sublabels
        `mean` and `standard deviation`. If `self.domain` is an instance of
        :class:`~nifty8.multi_domain.MultiDomain`, these sublabels refer
        themselves to subgroups. For :class:`~nifty8.field.Field`, the sublabel
        refers to an HDF5 data set.

        If quanitities are not requested (e.g. by setting `mean=False`), the
        respective sublabels are not present in the HDF5 file.

        If `op` is an :class:`~nifty8.operators.operator.Operator`, the operator
        string representation, its domain and its target are written to the
        attributes of the HDF5 file.

        Parameters
        ----------
        file_name : str
            File name of output hdf5 file.
        op : callable or None
            Callable that is applied to each item in the :class:`SampleListBase`
            before it is returned. Can be an
            :class:`~nifty8.operators.operator.Operator` or any other callable
            that takes a :class:`~nifty8.field.Field` as an input. Default:
            None.
        samples : bool
            If True, samples are written into hdf5 file.
        mean : bool
            If True, mean of samples is written into hdf5 file.
        std : bool
            If True, standard deviation of samples is written into hdf5 file.
        overwrite : bool
            If True, a potentially existing file with the same file name as
            `file_name`, is overwritten.
        """
        import h5py

        if os.path.isfile(file_name):
            if self.MPI_master and overwrite:
                os.remove(file_name)
            if not overwrite:
                raise RuntimeError(f"File {file_name} already exists. Delete it or use "
                                   "`overwrite=True`")
        if not (samples or mean or std):
            raise ValueError("Neither samples nor mean nor standard deviation shall be written.")

        if self.MPI_master:
            f = h5py.File(file_name, "w")
            if isinstance(op, Operator):
                f.attrs["nifty operator string representation"] = str(op)
                f.attrs["nifty operator domain"] = str(op.domain)
                f.attrs["nifty operator target"] = str(op.target)
        else:
            f = utilities.Nop()

        # TODO Add some meta information (e.g. k_length for PowerSpace, distances for RGSpace)
        if samples:
            grp = f.create_group("samples")
            for ii, ss in enumerate(self.iterator(op)):
                _field2hdf5(grp, ss, str(ii))
        if mean or std:
            grp = f.create_group("stats")
            m, v = self.sample_stat(op)
        if mean:
            _field2hdf5(grp, m, "mean")
        if std:
            _field2hdf5(grp, v.sqrt(), "standard deviation")

        f.close()
        _barrier(self.comm)

    def save_to_fits(self, file_name_base, op=None, samples=False, mean=False, std=False,
                     overwrite=False):
        """Write sample list to FITS file.

        This function writes properties of a sample list to a FITS file. This is
        supported if and only if the target of `op` is a two-dimensional
        RGSpace.

        Parameters
        ----------
        file_name_base : str
            File name base of output FITS file, i.e. without `.fits` extension.
        op : callable or None
            Callable that is applied to each item in the :class:`SampleListBase`
            before it is returned. Can be an
            :class:`~nifty8.operators.operator.Operator` or any other callable
            that takes a :class:`~nifty8.field.Field` as an input. Default:
            None.
        samples : bool
            If True, samples are written into hdf5 file.
        mean : bool
            If True, mean of samples is written into hdf5 file.
        std : bool
            If True, standard deviation of samples is written into hdf5 file.
        overwrite : bool
            If True, a potentially existing file with the same file name as
            `file_name`, is overwritten.
        """
        # TODO Add support for 3d and 4d fields
        # https://fits.gsfc.nasa.gov/fits_standard.html

        if not (samples or mean or std):
            raise ValueError("Neither samples nor mean nor standard deviation shall be written.")

        if mean or std:
            m, s = self.sample_stat(op)
        if mean:
            self._save_fits_2d(m, file_name_base + "_mean.fits", overwrite)
        if std:
            self._save_fits_2d(s, file_name_base + "_std.fits", overwrite)

        if samples:
            for ii, ss in enumerate(self.iterator(op)):
                self._save_fits_2d(ss, file_name_base + f"_sample_{ii}.fits", overwrite)

    def _save_fits_2d(self, fld, file_name, overwrite):
        import astropy.io.fits as pyfits
        from astropy.time import Time

        dom = fld.domain
        if len(dom) != 1 or len(dom[0].shape) != 2:
            raise ValueError("FITS file export is only supported from 2d-fields. "
                             f"Current domain:\n{dom}")
        h = pyfits.Header()
        h["DATE-MAP"] = Time(time.time(), format="unix").iso.split()[0]
        h["CRVAL1"] = h["CRVAL2"] = 0
        h["CRPIX1"] = h["CRPIX2"] = 0
        h["CUNIT1"] = h["CUNIT2"] = "deg"
        h["CDELT1"], h["CDELT2"] = -dom[0].distances[0], dom[0].distances[1]
        h["CTYPE1"] = "RA---SIN"
        h["CTYPE2"] = "DEC---SIN"
        h["EQUINOX"] = 2000

        hdu = pyfits.PrimaryHDU(fld.val[:, :].T, header=h)
        hdulist = pyfits.HDUList([hdu])
        if self.MPI_master:
            hdulist.writeto(file_name, overwrite=overwrite)

    def iterator(self, op=None):
        """Return iterator over all potentially distributed samples.

        Parameters
        ----------
        op : callable or None
            Callable that is applied to each item in the :class:`SampleListBase`
            before it is returned. Can be an
            :class:`~nifty8.operators.operator.Operator` or any other callable
            that takes a :class:`~nifty8.field.Field` as an input. Default:
            None.

        Note
        ----
        Calling this function involves MPI communication if `comm != None`.
        """
        op = _none_to_id(op)
        if self.comm is not None:
            for itask in range(self.comm.Get_size()):
                for i in range(_bcast(self.n_local_samples, self._comm, itask)):
                    ss = self.local_item(i) if itask == self._comm.Get_rank() else None
                    yield op(_bcast(ss, self._comm, itask))
        else:
            for ss in self.local_iterator():
                yield op(ss)

    def average(self, op=None):
        """Compute average over all potentially distributed samples.

        Parameters
        ----------
        op : callable or None
            Callable that is applied to each item in the :class:`SampleListBase`
            before it is averaged. If `op` returns tuple, then individual
            averages are computed and returned individually as tuple.

        Note
        ----
        Calling this function involves MPI communication if `comm != None`.

        Note
        ----
        Calling this function involves allocating arrays for all samples and
        averaging them afterwards. If the number of local samples is big and
        `op` is not None, this leads to much temporary memory usage. If the
        output of `op` is just a :class:`~nifty8.field.Field` or
        :class:`~nifty8.multi_field.MultiField`, :attr:`sample_stat()`
        can be used in order to compute the average memory efficiently.
        """
        op = _none_to_id(op)
        res = [op(ss) for ss in self.local_iterator()]
        n = self.n_samples()
        if not isinstance(res[0], tuple):
            return utilities.allreduce_sum(res, self.comm) / n
        n_output_elements = len(res[0])
        res = [[elem[ii] for elem in res] for ii in range(n_output_elements)]
        return tuple(utilities.allreduce_sum(rr, self.comm) / n for rr in res)

    def sample_stat(self, op=None):
        """Compute mean and variance of samples after applying `op`.

        Parameters
        ----------
        op : callable or None
            Callable that is applied to each item in the :class:`SampleListBase`
            before it is used to compute mean and variance.

        Returns
        -------
        tuple
            A tuple with two items: the mean and the variance.
        """
        from ..probing import StatCalculator
        sc = StatCalculator()
        for ss in self.iterator(op):
            sc.add(ss)
        return sc.mean, sc.var

    def save(self, file_name_base, overwrite=False):
        """Serialize SampleList and write it to disk.

        Parameters
        ----------
        file_name_base : str
            File name of the output file without extension. The actual file name
            will have the extension ".pickle" and before that an identifier that
            distunguishes between MPI tasks.
        overwrite : bool
            Existing files are overwritten.

        Note
        ----
        If the instance of :class:`SampleListBase` is distributed, each MPI task
        writes its own file.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, file_name_base, comm=None):
        """Deserialize SampleList from files on disk.

        Parameters
        ----------
        file_name_base : str
            File name of the input file without extension. The actual file name
            will have the extension ".pickle" and before that an identifier that
            distunguishes between MPI tasks.
        comm : MPI communicator or None
            If not `None`, each MPI task reads its own input file.

        Note
        ----
        `file_name_base` needs to be the same string that has been used for
        saving the :class:`SampleListBase`.

        Note
        ----
        The number of MPI tasks used for saving and loading the `SampleList`
        need to be the same.
        """
        raise NotImplementedError

    @classmethod
    def _list_local_sample_files(cls, file_name_base, comm=None):
        """List all sample files that are relevant for the local task.

        All sample files that correspond to `file_name_base` are searched and
        selected based on which rank the local task has.

        Note
        ----
        This function makes sure that the all file numbers between 0 and the
        maximal found number are present. If this is not the case, a
        `RuntimeError` is raised.
        """
        base_dir, base_file = os.path.split(os.path.abspath(file_name_base))
        files = [ff for ff in os.listdir(base_dir)
                 if re.match(f"{base_file}.[0-9]+.pickle", ff) ]
        if len(files) == 0:
            raise RuntimeError(f"No files matching `{file_name_base}.*.pickle`")
        n_samples = max(list(map(lambda x: int(x.split(".")[-2]), files))) + 1
        files = [f"{file_name_base}.{ii}.pickle" for ii in cls.local_indices(n_samples, comm)]
        for ff in files:
            if not os.path.isfile(ff):
                raise RuntimeError(f"File {ff} not found")
        return files


class ResidualSampleList(SampleListBase):
    def __init__(self, mean, residuals, neg, comm=None):
        """Store samples in terms of a mean and a residual deviation thereof.


        Parameters
        ----------
        mean : Field or MultiField
            Mean of the sample list.
        residuals : list of Field or list of MultiField
            List of residuals from the mean. If it is a list of `MultiField`,
            the domain of the residuals can be a subdomain of the domain of
            mean. This results in adding just a zero in respective `MultiField`
            entries.
        neg: list of bool
            This list has to have the same length as `residuals`. If an entry is
            `True`, the respective residual is subtracted and not added.
        comm : MPI communicator or None
            If not `None`, samples can be gathered across multiple MPI tasks. If
            `None`, :class:`ResidualSampleList` is not a distributed object.
        """
        super(ResidualSampleList, self).__init__(comm, mean.domain)
        self._m = mean
        self._r = tuple(residuals)
        self._n = tuple(neg)

        if len(self._r) != len(self._n):
            raise ValueError("Residuals and neg need to have the same length.")

        r_dom = self._r[0].domain
        if not all(rr.domain is r_dom for rr in self._r):
            raise ValueError("All residuals must have the same domain.")
        if isinstance(r_dom, MultiDomain):
            try:
                self._m.extract(r_dom)
            except:
                raise ValueError("`residual.domain` must be a subdomain of `mean.domain`.")

        if not all(isinstance(nn, bool) for nn in neg):
            raise TypeError("All entries in neg need to be bool.")

    def local_item(self, i):
        return self._m.flexible_addsub(self._r[i], self._n[i])

    @property
    def n_local_samples(self):
        return len(self._r)

    def at(self, mean):
        """Instantiate `ResidualSampleList` with the same residuals as `self`.

        The mean is updated.

        Note
        ----
        If `self.domain` is a :class:`~nifty8.multi_domain.MultiDomain`, the old
        and new mean are combined with
        :attr:`~nifty8.multi_field.MultiField.union` beforehand. This means that
        only the multi field entries present in `mean` are updated.

        Returns
        -------
        ResidualSampleList
            Sample list with updated mean.
        """
        if isinstance(self._m, MultiField) and self.domain is not mean.domain:
            mean = MultiField.union([self._m, mean])
        return ResidualSampleList(mean, self._r, self._n, self.comm)

    def save(self, file_name_base, overwrite=False):
        nsample = self.n_samples()
        local_indices = self.local_indices(nsample, self.comm)
        for ii, isample in enumerate(local_indices):
            obj = [self._r[ii], self._n[ii]]
            fname = _sample_file_name(file_name_base, isample)
            _save_to_disk(fname, obj, overwrite)
        if self.MPI_master:
            _save_to_disk(f"{file_name_base}.mean.pickle", self._m, overwrite)
        _barrier(self.comm)

    @classmethod
    def load(cls, file_name_base, comm=None):
        _barrier(comm)
        mean = _load_from_disk(f"{file_name_base}.mean.pickle")
        files = cls._list_local_sample_files(file_name_base, comm)
        tmp = [_load_from_disk(ff) for ff in files]
        res = [aa[0] for aa in tmp]
        neg = [aa[1] for aa in tmp]
        return cls(mean, res, neg, comm=comm)

    @classmethod
    def load_mean(cls, file_name_base):
        return _load_from_disk(f"{file_name_base}.mean.pickle")


class SampleList(SampleListBase):
    def __init__(self, samples, comm=None, domain=None):
        """Store samples as a plain list.

        This is a minimalist implementation of :class:`SampleListBase`. It just
        serves as a (potentially MPI-distributed) wrapper of a list of samples.

        Parameters
        ----------
        samples : list of Field or list of MultiField
            List of samples.
        comm : MPI communicator or None
            If not `None`, samples can be gathered across multiple MPI tasks. If
            `None`, :class:`ResidualSampleList` is not a distributed object.
        domain : DomainTuple, MultiDomain or None
            Sets the domain of the `SampleList`. If `samples` is non-empty and
            `domain` is not None, `domain` has to coincide with the domain of
            the samples. Default: None.
        """
        from ..sugar import makeDomain

        if domain is None:
            if len(samples) == 0:
                raise ValueError("Need to pass `domain` to instantiate empty `SampleList`.")
            else:
                domain = samples[0].domain
        else:
            domain = makeDomain(domain)
        super(SampleList, self).__init__(comm, domain)
        self._s = samples
        for ss in self._s:
            utilities.check_object_identity(ss.domain, self.domain)

    def local_item(self, i):
        return self._s[i]

    @property
    def n_local_samples(self):
        return len(self._s)

    def save(self, file_name_base, overwrite=False):
        nsample = self.n_samples()
        local_indices = self.local_indices(nsample, self.comm)
        for isample in range(nsample):
            if isample in local_indices:
                obj = self._s[isample-local_indices[0]]
                fname = _sample_file_name(file_name_base, isample)
                _save_to_disk(fname, obj, overwrite=True)
        _barrier(self.comm)

    @classmethod
    def load(cls, file_name_base, comm=None):
        _barrier(comm)
        files = cls._list_local_sample_files(file_name_base, comm)
        samples = [_load_from_disk(ff) for ff in files]
        return cls(samples, comm=comm)


def _none_to_id(obj):
    """If the input is None, replace it with identity map. If input is
    `Operator`, append `force` in order to make sure that the function is
    evaluated if there is a chance that the domains match. Otherwise return
    input.
    """
    if obj is None:
        return lambda x: x
    if isinstance(obj, Operator):
        return lambda x: obj.force(x)
    return obj


def _bcast(obj, comm, root):
    """Broadcast python object from given root.

    Parameters
    ----------
    obj : object
        The object to be broadcasted.
    comm : MPI communicator
        MPI communicator used for the broadcasting.
    root : int
        MPI task number from which the object shall be sent.
    """
    data = obj if comm.Get_rank() == root else None
    return comm.bcast(data, root=root)


def _sample_file_name(file_name_base, isample):
    """Return sample-unique file name.

    This file name that can be used to uniquely write samples potentially from all MPI tasks.

    Parameters
    ----------
    file_name_base : str
        User-defined first part of file name.
    isample : int
        Number of sample
    """
    if not isinstance(isample, int):
        raise TypeError
    return f"{file_name_base}.{isample}.pickle"


def _load_from_disk(file_name):
    with open(file_name, "rb") as f:
        obj = pickle.load(f)
    return obj


def _save_to_disk(file_name, obj, overwrite=False):
    if overwrite and os.path.isfile(file_name):
        os.remove(file_name)
    with open(file_name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def _field2hdf5(file_handle, obj, name):
    if not isinstance(name, str):
        raise TypeError
    if isinstance(obj, MultiField):
        grp = file_handle.create_group(name)
        for kk, fld in obj.items():
            _field2hdf5(grp, fld, kk)
        return
    if not isinstance(obj, Field):
        raise TypeError
    file_handle.create_dataset(name, data=obj.val)


def _barrier(comm):
    if comm is not None:
        comm.Barrier()
