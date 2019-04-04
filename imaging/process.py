# -*- coding: utf-8 -*-

import numpy as np
from skimage.segmentation import checkerboard_level_set, morphological_chan_vese
from skimage.measure import label, regionprops



class process(object):
    def __init__(self):
        pass
    
    def threshold(self, image, threshold_value):
        output_image = image*1
        output_image =  np.where(image < threshold_value, np.min(image), np.max(image))
        return output_image
    
    def compute_roi(self, image):
        init_ls = checkerboard_level_set(image_shape=image.shape, square_size=15)
        detected_rois = morphological_chan_vese(image=image, iterations=35, init_level_set=init_ls, smoothing=0)
        labels = label(detected_rois)
        regions = regionprops(labels)
        return detected_rois, labels, regions