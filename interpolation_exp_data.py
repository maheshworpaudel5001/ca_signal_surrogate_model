#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:00:07 2024

@author: ixn004
"""


import numpy as np
import scipy as sp
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d


def exp_idata(time,exp_data,tnew):
    g=sp.interpolate.make_interp_spline(time,exp_data)
    #g=sp.interpolate.interp1d(time,exp_data,kind='cubic',bounds_error=False)
    idata=g(tnew) #type
    return idata,tnew