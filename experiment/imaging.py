import numpy as np
import skimage
from nd2reader import ND2Reader
import pandas as pd
class imaging(object):
    def __init__(self):
        pass

    def rois(self, filename):
        self.roi_mask = skimage.data.imread(filename)
        self.labels = skimage.measure.label(self.roi_mask)
        self.regions = skimage.measure.regionprops(self.labels)
        return self.regions

    def video(self, filename):
        self.frames = []
        images = ND2Reader(filename)
        self.intensities = np.zeros((len(images), len(self.regions)))
        self.time = np.reshape((images.timesteps - images.timesteps[0]), [1, len(images)])
        for image in images:
            self.frames.append(image*self.roi_mask)
        images.close()
        return self.frames

    def responses(self):
        for f in range(0, len(self.frames)):
            frame_regions = skimage.measure.regionprops(self.labels, intensity_image=self.frames[f])
            for r in range(0, len(frame_regions)):
                self.intensities[f, r] = frame_regions[r].mean_intensity
        return self.intensities

    def _to_DataFrame(self, normalize=False):
        data = pd.DataFrame(self.intensities)
        if normalize == True:
            data = (data - data.mean()) / (data.max() - data.min())
        return data