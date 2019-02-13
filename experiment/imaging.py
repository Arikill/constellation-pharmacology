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
        self.areas[0, region_index] = self.regions[0, region_index].area
        pass
    
    def compute_intensities(self, index):
        frame_regions_340 = skimage.measure.regionprops(self.labels, intensity_image=self.frames_340[index, :, :])
        frame_regions_380 = skimage.measure.regionprops(self.labels, intensity_image=self.frames_380[index, :, :])
        for r in range(0, len(frame_regions_340)):
            self.intensities_340[index, r] = frame_regions_340[r].mean_intensity
            self.intensities_380[index, r] = frame_regions_380[r].mean_intensity
        print(index)
        pass

    def rois(self, filename):
        self.roi_mask = skimage.img_as_bool(skimage.data.imread(filename))
        self.labels = skimage.measure.label(self.roi_mask)
        self.regions = np.asarray(skimage.measure.regionprops(self.labels), dtype=np.object)
        self.nregions = len(self.regions)
        self.regions = np.reshape(self.regions, [1, self.nregions])
        self.areas = np.zeros((1, self.nregions))
        Parallel(n_jobs=-1, require='sharedmem')(delayed(self.append_roi_areas)(region_index=r) for r in range(self.nregions))
        return self.regions

    def video(self, filename):
        self.sampling_rate = 0.5
        images = ND2Reader(filename)
        rows, cols = images[0].shape
        self.nframes = len(images)
        self.frames_340 = np.zeros((self.nframes, rows, cols))
        self.frames_380 = np.zeros((self.nframes, rows, cols))
        self.intensities_340 = np.zeros((self.nframes, self.nregions))
        self.intensities_380 = np.zeros((self.nframes, self.nregions))
        for image_index in range(self.nframes):
            current_image = images[image_index]
            current_image.iter_axes('c')
            self.frames_340[image_index, :, :] += current_image[0]*self.roi_mask
            self.frames_380[image_index, :, :] += current_image[1]*self.roi_mask
            print(image_index)
        Parallel(n_jobs=-1, require='sharedmem')(delayed(self.compute_intensities)(index=i) for i in range(self.nframes))
        self.time = np.reshape((images.timesteps - images.timesteps[0]), [1, self.nframes])
        images.close()
        return self.frames
    
    def filter(self, cut_off, order):
        nyq_rate = self.sampling_rate/2
        cutoff = cut_off/nyq_rate
        b, a = signal.butter(N=order, Wn=cutoff, btype="low")
        rows, cols = self.intensities.shape
        for col in range(cols):
            self.intensities[:, col] = signal.filtfilt(b, a, self.intensities[:, col])
        pass

    def _to_DataFrame(self, normalize=False):
        data_340 = pd.DataFrame(self.intensities_340)
        data_380 = pd.DataFrame(self.intensities_380)
        if normalize == True:
            data_340 = (data_340 - data_340.mean()) / (data_340.max() - data_340.min())
            data_380 = (data_380 - data_380.mean()) / (data_380.max() - data_380.min())
        return data_340, data_380
    
    def categorize(self, algorithm, parameters, normalize=True):
        if normalize == True:
            data_340, data_380 = self._to_DataFrame(normalize=True)
            intensities_340 = data_340.get_values()
            intensities_380 = data_380.get_values()
#            rows, cols = intensities.shape
#            for col in range(cols):
#                mean = np.mean(intensities[:, col])
#                maximum = np.max(intensities[:, col])
#                minimum = np.min(intensities[:, col])
#                intensities[:, col] = (intensities[:, col]-mean)/(maximum-minimum)
        else:
            intensities_340 = self.intensities_340
            intensities_380 = self.intensities_380
        if algorithm == "kmeans":
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=parameters["n_clusters"])
            kmeans.fit(intensities.T)
#            centroids = kmeans.cluster_centers_
            self.categories = np.reshape(kmeans.labels_, [1, self.nregions])
        if algorithm == "correlation":
            corr_340 = np.corrcoef(self.intensities_340.T)
            corr_380 = np.corrcoef(self.intensities_380.T)
            rows, cols = corr_340.shape
            self.categories = np.ones((1, self.nregions))+np.inf
            for col in range(cols):
                for row in range(col+1, rows):
                    if corr[row][col] >= parameters["threshold"]:
                        if self.categories[0, row] >= np.inf:
                            self.categories[0, row] = col
        pass
            
            
            
            