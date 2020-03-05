# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.


def _logger_init():
    import logging
    res = logging.getLogger('NIFTy6')
    res.setLevel(logging.DEBUG)
    res.propagate = False
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    res.addHandler(ch)
    return res


logger = _logger_init()
