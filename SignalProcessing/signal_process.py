#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 23:11:43 2019

@author: rishialluri
"""

from scipy.signal import butter, filtfilt
from scipy.sparse import spdiags, csc_matrix, linalg
import numpy as np

class signal_process(object):
    def __init__(self):
        pass
    
    def filter_intensity(self, intensity, rate, cutoff, order):
        nyquist_rate = rate/2
        nyquist_cutoff = cutoff/nyquist_rate
        b, a = butter(N=order, Wn=nyquist_cutoff, btype="low")
        return filtfilt(b, a, intensity)
        
    def remove_drift(self, intensity, lam=10**7, p=0.01, iterations=50):
        samples = len(intensity)
        D = csc_matrix(np.diff(np.eye(samples), 2))
        w = np.ones(samples)
        for i in range(iterations):
            W = spdiags(w, 0, samples, samples)
            Z = W + lam * D.dot(D.transpose())
            drift = linalg.spsolve(Z, w*intensity)
            w = p * (intensity > drift) + (1-p) * (intensity < drift)
        return intensity-drift+intensity[0]
        
    
    def filter_intensities(self, intensities, rate, cutoff, order):
        y = intensities*0
        nyquist_rate = rate/2
        nyquist_cutoff = cutoff/nyquist_rate
        b, a = butter(N=order, Wn=nyquist_cutoff, btype="low")
        rows, cols = y.shape
        for region in range(rows):
            y[region, :] = filtfilt(b, a, intensities[region, :])
            print("Filtering: ", region)
        return y
    
    def remove_drifts(self, intensities, lam=10**5, p=0.01, iterations=10):
        y = intensities * 0
        rows, cols = y.shape
        for region in range(rows):
            D = csc_matrix(np.diff(np.eye(cols), 2))
            w = np.ones(cols)
            for i in range(iterations):
                W = spdiags(w, 0, cols, cols)
                Z = W + lam * D.dot(D.transpose())
                drift = linalg.spsolve(Z, w*intensities[region, :])
                y[region, :] = intensities[region, :]-drift+intensities[region, 0]
            print("Removing drift: ", region)
        return y