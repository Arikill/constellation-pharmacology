# -*- coding: utf-8 -*-

from nd2reader import ND2Reader
import numpy as np
from joblib import Parallel, delayed
from skimage import img_as_bool, measure, img_as_uint
from skimage.external.tifffile import imread
from SignalProcessing.signal_process import signal_process
sig_proc = signal_process()

class reader(object):
    
    def __init__(self):
        self.progress = 0
        pass
    
    def set_roi_file(self, roi_file):
        self.roi_file = roi_file
        self.mask = img_as_bool(imread(self.roi_file))
        self.labels = measure.label(self.mask)
        self.regions = measure.regionprops(self.labels)
        self.nregions = len(self.regions)
        pass
    
    def set_imaging_file(self, imaging_file):
        self.imaging_file = imaging_file
        pass
    
    def set_start_time(self, start_time):
        self.start_time = start_time
        pass

    def set_end_time(self, end_time):
        self.end_time

    def read(self, roi_file=False, imaging_file=False, start_time=False, end_time=False):
        if roi_file:
            self.set_roi_file(roi_file)
        else:
            ValueError("No roi file found!")
        if imaging_file:
            self.set_imaging_file(imaging_file)
        else:
            ValueError("No imaging file found!")
        images = ND2Reader(self.imaging_file)
        self.ypixels, self.xpixels = images.frame_shape
        self.rate = images.frame_rate
        self.nframes = images.sizes['t']
        if end_time:
            self.set_end_time(end_time)
            end_frame = np.floor(self.rate*self.end_time)
        else:
            end_frame = self.nframes
            self.end_time = np.floor(end_frame/self.rate)
            
        if start_time:
            self.set_start_time(start_time)
            start_frame = np.floor(self.rate*self.start_time)
        else:
            start_frame = 0
            self.start_time = np.floor(start_frame/self.rate)
        self.intensities = np.zeros((self.nregions, end_frame-start_frame), dtype=np.float32)
        self.time = np.linspace(self.start_time, self.end_time, end_frame-start_frame)
        for frame_index in range(start_frame, end_frame):
            frame_340 = images.get_frame_2D(t=frame_index, c=0)*self.mask
            frame_380 = images.get_frame_2D(t=frame_index, c=1)*self.mask
            ratio = np.where(frame_380 == 0, 0, frame_340/frame_380)
            regions = measure.regionprops(label_image=self.labels, intensity_image=ratio)
            for region_index in range(self.nregions):
                self.intensities[region_index, frame_index-start_frame] = regions[region_index].mean_intensity
            self.progress = np.round(((frame_index-start_frame)/(end_frame-start_frame))*100, 2)
            print(str(self.progress),"%")
        images.close()
        pass
        
    def normalize(self, matrix):
        normalized_matrix = (matrix - np.min(matrix))/(np.max(matrix) - np.min(matrix))
        return normalized_matrix
    
    def construct_roi(self, imaging_file=False, ftfn = False):
        if imaging_file:
            self.set_imaging_file = imaging_file
        images = ND2Reader(self.set_imaging_file)
        self.nframes = images.sizes['t']
        self.rate = images.frame_rate
        rows, cols = images.frame_shape
        self.differential = np.zeros((rows, cols), dtype=np.float32)
        frame_340 = images.get_frame_2D(t=0, c=0)
        frame_380 = images.get_frame_2D(t=0, c=1)
        if ftfn:
            past_ratio = self.normalize(np.where(frame_380 == 0, 0, frame_340/380))
        past_ratio = np.where(frame_380 == 0, 0, frame_340/380)
        for frame_index in range(1, self.nframes):
            frame_340 = images.get_frame_2D(t=frame_index, c=0)
            frame_380 = images.get_frame_2D(t=frame_index, c=1)
            if ftfn:
                current_ratio = self.normalize(np.where(frame_380 == 0, 0, frame_340/380))
            current_ratio = np.where(frame_380 == 0, 0, frame_340/380)
            self.differential += current_ratio - past_ratio
            past_ratio = current_ratio
            self.progress = np.round((frame_index/self.nframes)*100, 2)
        images.close()
        return self.differential*self.rate

    def correct_intensities(self, parallel=False):
        self.corrected_intensities = self.intensities*0
        if not parallel:
            for region_index in range(self.nregions):
                drift_removed = sig_proc.remove_region_drift(intensity=self.intensities[region_index, :], lam=10**7, p=0.01, iterations=50)
                self.corrected_intensities[region_index, :] = sig_proc.filter_region_intensity(intensity=drift_removed, rate=self.rate, cutoff=0.1, order=3)
                self.progress = np.round((region_index/self.nregions)*100, 2)
                print(str(self.progress),"%")
        else:
            self.index_count = 0
            Parallel(n_jobs=10, require="sharedmem", prefer="threads")(delayed(self.correct_region_intensity)(region_index=r)for r in range(self.nregions))
        pass

    def correct_region_intensity(self, region_index):
        drift_removed = sig_proc.remove_region_drift(intensity=self.intensities[region_index, :], lam=10**7, p=0.01, iterations=50)
        self.corrected_intensities[region_index, :] = sig_proc.filter_region_intensity(intensity=drift_removed, rate=self.rate, cutoff=0.1, order=3)
        self.index_count += 1
        self.progress = np.round((self.index_count/self.nregions)*100, 2)
        print(str(self.progress),"%")
        pass
    
    def extract_cell_type(self, number):
        self.correlation = np.corrcoef(self.corrected_intensities)
        pass
    