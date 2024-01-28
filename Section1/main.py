#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:33:21 2024

@author: chiahunglee
"""

from function import *
# test code

data_fls, ref_fls = LoadTroikaDataset()

errors, confidence = RunPulseRateAlgorithm(data_fls[0], ref_fls[0])
