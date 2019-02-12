from experiment.imaging import imaging
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# ... do stuff

roi_file = "Z:/Lee Leavitt/190208.29.m.m3.p1 pl14a testing/roi.1024.tif"
video_file = "Z:/Lee Leavitt/190208.29.m.m3.p1 pl14a testing/video_3007.nd2"

import time
start_time = time.time()
img = imaging()
img.rois(roi_file)
img.video(video_file)
end_time = time.time()
print("Elapsed time was %g seconds" % (end_time - start_time))
img.responses()

#img.filter(0.21, 3)
data = img._to_DataFrame(normalize=False)

parameters = {}
parameters["threshold"] = 0.93
img.categorize("correlation", parameters, True)
cat = img.categories

figure1 = plt.figure()
axes1 = figure1.add_subplot(111, projection='3d')
axes1.scatter(data.columns, img.categories, img.areas)
axes1.set_xlabel("Cells")
axes1.set_ylabel("Type")
axes1.set_zlabel("Pixels")


figure2 = plt.figure()
axes2 = figure2.add_subplot(111)
cell_type = 7
_, cells = np.where(cat == cell_type)
for cell in cells:
    axes2.plot(img.time[0, :], data[cell])
    
    