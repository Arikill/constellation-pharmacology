from experiment.imaging import imaging
import sklearn
roi_file = "data/roi1024.tif"
video_file = "data/video038.nd2"

img = imaging()
img.rois(roi_file)
img.video(video_file)
img.responses()
data = img._to_DataFrame()
clusters = sklearn.cluster.KMeans(n_clusters=8)