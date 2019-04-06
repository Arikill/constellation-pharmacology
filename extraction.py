#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 21:45:02 2019

@author: rishialluri
"""

from imaging.reader import reader
from matplotlib import pyplot as plt
from SignalProcessing.signal_process import signal_process
import numpy as np
import time

rdr = reader()
sig_proc = signal_process()

video_file = "Z:/Halen/Ephyz/190322.m.40.m3.p1/video_1001.nd2"
roi_file = "Z:/Halen/Ephyz/190322.m.40.m3.p1/roi1024.tif"

start_time = time.time()
rdr.read(imaging_file=video_file, roi_file=roi_file)
print("--- Reading: %s seconds ---" % (time.time() - start_time))
start_time = time.time()
rdr.correct_intensities(parallel=True)
print("--- Correcting: %s seconds ---" % (time.time() - start_time))