import numpy as np
import skimage
from nd2reader import ND2Reader
import pandas as pd
import scipy.signal as signal
from joblib import Parallel, delayed

class imaging(object):
    def __init__(self):
        pass
    
    def append_roi_areas(self, region_index):
        self.areas[0, region_index] = self.regions[region_index].area
        pass
    
    def append_modified_frames(self, image_index):
        self.frames[:, :, image_index] *= self.roi_mask
        pass
    
    def compute_intensities(self, frame_index):
        frame_regions = skimage.measure.regionprops(self.labels, intensity_image=self.frames[:, :, frame_index])
        for r in range(0, len(frame_regions)):
            self.intensities[frame_index, r] = frame_regions[r].mean_intensity
        pass
        

    def rois(self, filename):
        self.roi_mask = skimage.img_as_bool(skimage.data.imread(filename))
        self.labels = skimage.measure.label(self.roi_mask)
        self.regions = skimage.measure.regionprops(self.labels)
        self.areas = np.zeros((1, len(self.regions)))
        Parallel(n_jobs=-2, require='sharedmem')(delayed(self.append_roi_areas)(region_index=r) for r in range(0, len(self.regions)))
        return self.regions

    def video(self, filename):
        self.sampling_rate = 0.5
        self.frames = []
        images = ND2Reader(filename)
        self.intensities = np.zeros((len(images), len(self.regions)))
        self.time = np.reshape((images.timesteps - images.timesteps[0]), [1, len(images)])
        rows, cols = images[0].shape
        nframes = len(images)
        images.close()
        self.frames = np.zeros((rows, cols, nframes), dtype=np.uint8)
        Parallel(n_jobs=-2, require='sharedmem')(delayed(self.append_modified_frames)(image_index=i) for i in range(0, nframes))
        return self.frames

    def responses(self):
        Parallel(n_jobs=-2, require='sharedmem')(delayed(self.compute_intensities)(frame_index=i) for i in range(0, self.frames.shape[2]))
        pass
    
    def filter(self, cut_off, order):
        nyq_rate = self.sampling_rate/2
        cutoff = cut_off/nyq_rate
        b, a = signal.butter(N=order, Wn=cutoff, btype="low")
        rows, cols = self.intensities.shape
        for col in range(cols):
            self.intensities[:, col] = signal.filtfilt(b, a, self.intensities[:, col])
        pass

    def _to_DataFrame(self, normalize=False):
        data = pd.DataFrame(self.intensities)
        if normalize == True:
            data = (data - data.mean()) / (data.max() - data.min())
        return data
    
    def categorize(self, algorithm, parameters, normalize=True):
        if normalize == True:
            data = self._to_DataFrame(normalize=True)
            intensities = data.get_values()
            rows, cols = intensities.shape
            for col in range(cols):
                mean = np.mean(intensities[:, col])
                maximum = np.max(intensities[:, col])
                minimum = np.min(intensities[:, col])
                intensities[:, col] = (intensities[:, col]-mean)/(maximum-minimum)
        else:
            intensities = self.intensities
        if algorithm == "kmeans":
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=parameters["n_clusters"])
            kmeans.fit(intensities.T)
#            centroids = kmeans.cluster_centers_
            self.categories = np.reshape(kmeans.labels_, [1, len(self.regions)])
        if algorithm == "correlation":
            corr = np.corrcoef(self.intensities.T)
            rows, cols = corr.shape
            self.categories = np.ones((1, len(self.regions)))+np.inf
            for col in range(cols):
                for row in range(col+1, rows):
                    if corr[row][col] >= parameters["threshold"]:
                        if self.categories[0, row] >= np.inf:
                            self.categories[0, row] = col
        pass
            
            
            
            