import numpy as np
from joblib import Parallel, delayed
class processing(object):
    def __init__(self):
        pass
    
    def compute_ratio(self, frames):
        print("creating frames ration")
        frames_ratio = np.zeros(frames[340].shape, dtype=np.uint16)
        nframes = frames_ratio.shape[0]
        print("Parallel Computing Frames for ", nframes)
        def ratio(frame_index):
            print("Entered ratio function")
            frames_ratio[frame_index, :, :] = frames[340][frame_index, :, :]/frames[380][frame_index, :, :]
            print("Computed ratio at ", frame_index, "|", nframes)
            pass
        Parallel(n_jobs=-1)(delayed(ratio)(frame_index=i) for i in range(0, nframes))
        return frames_ratio
    
    def compute_difference(self, frames):
        frames_difference = frames["ratio"][0:-2, :, :]
        nframes = frames_difference.shape[0]
        for frame_index in range(nframes):
            frames_difference[frame_index, :, :] = frames["ratio"][frame_index+1, :, :] - frames["ratio"][frame_index, :, :]
            print("Computed difference at ", frame_index, "|", nframes)
        return frames_difference