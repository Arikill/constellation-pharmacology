from skimage import measure, img_as_bool, data
import numpy as np
from nd2reader import ND2Reader
from joblib import Parallel, delayed
class file_operations(object):
    def __init__(self):
        pass

    def read_rois(self, filename):
        self.mask = img_as_bool(data.imread(filename))
        self.labels = measure.label(self.mask)
        self.regions = np.asarray(measure.regionprops(self.labels), dtype=np.object)
        self.nregions = len(self.regions)
        self.areas = np.zeros((1, self.nregions), dtype=np.float32)
        pass
    
    def read_video(self, filename):
        self.sampling_rate = 0.5
        self.frames = {}
        self.intensities = {}
        images = ND2Reader(filename)
        self.nframes = len(images)
        x, y = images[0].shape
        self.frames[340] = np.zeros((self.nframes, x, y), dtype=np.uint16)
        self.frames[380] = np.zeros((self.nframes, x, y), dtype=np.uint16)
        self.frames["ratio"] = np.zeros((self.nframes, x, y), dtype=np.float32)
        for frame_index in range(self.nframes):
            image = images[frame_index]
            image.iter_axes = 'c'
            self.frames[340][frame_index, :, :] += np.uint16(image[0])
            self.frames[380][frame_index, :, :] += np.uint16(image[1])
            self.frames["ratio"][frame_index, :, :] += self.frames[340][frame_index, :, :]/self.frames[380][frame_index, :, :]
            print("Processed ", frame_index, "|", self.nframes)
        images.close()
        pass