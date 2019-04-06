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
    
    def filter_region_intensity(self, intensity, rate, cutoff, order):
        nyquist_rate = rate/2
        nyquist_cutoff = cutoff/nyquist_rate
        b, a = butter(N=order, Wn=nyquist_cutoff, btype="low")
        return filtfilt(b, a, intensity)
        
    def remove_region_drift(self, intensity, lam=10**7, p=0.01, iterations=50):
        samples = len(intensity)
        D = csc_matrix(np.diff(np.eye(samples), 2))
        w = np.ones(samples)
        for i in range(iterations):
            W = spdiags(w, 0, samples, samples)
            Z = W + lam * D.dot(D.transpose())
            drift = linalg.spsolve(Z, w*intensity)
            w = p * (intensity > drift) + (1-p) * (intensity < drift)
        return intensity-drift+intensity[0]
    
    def correlation_cluster(self, matrix, threshold):
        cluster = []
        rows, cols = matrix.shape
        correlation = np.corrcoef(matrix)
        cells = np.linspace(0, rows-1, rows, dtype=np.int).tolist()
        for row in cells:
            sub_cluster = np.asarray(np.where(correlation[row, :] >= threshold)).tolist()[0]
            for item in sub_cluster:
                correlation[item, :] = 0
            if not sub_cluster in cluster:
                cluster.append(sub_cluster)
            print(str((row/rows)*100), "%")
        return [x for x in cluster if x]
                