# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:25:06 2019

@author: DarthRishi
"""

from SignalProcessing.signal_process import signal_process
sig_proc = signal_process()

clusters = sig_proc.correlation_cluster(rdr.corrected_intensities, 0.92)
clusters.sort(key=len, reverse=True)
#
traces = clusters[31]
fig, ax = plt.subplots(len(traces), 1, sharex=True)
for trace in range(len(traces)):
    ax[trace].plot(rdr.time, rdr.corrected_intensities[traces[trace], :])
    ax[trace].set_ylabel("Cell: "+str(traces[trace]))
ax[len(traces)-1].set_xlabel("time (sec)")