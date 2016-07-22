# -*- coding: utf-8 -*-

from nifty.config import about
from nifty.spaces.space import SpaceParadict


class HPSpaceParadict(SpaceParadict):

    def __init__(self, nside):
        SpaceParadict.__init__(self, nside=nside)

    def __setitem__(self, key, arg):
        if key not in ['nside']:
            raise ValueError(about._errors.cstring(
                "ERROR: Unsupported hp_space parameter"))

        temp = int(arg)
        # if(not hp.isnsideok(nside)):
        if ((temp & (temp - 1)) != 0) or (temp < 2):
            raise ValueError(about._errors.cstring(
                "ERROR: invalid parameter ( nside <> 2**n )."))
        self.parameters.__setitem__(key, temp)
