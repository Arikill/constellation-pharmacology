import scipy.signal as scipy_signal
class signals(object):
    def __init__(self):
        pass

    def butterworth_filter(self, X, Fs, axis, cutoff, order):
        nyquist_rate = Fs/2
        nyquist_cutoff = cutoff/nyquist_rate
        b, a = scipy_signal.butter(N=order, Wn=nyquist_cutoff, btype="low")
        rows, cols = X.shape
        Y = X*0
        if axis == 0:
            for row in range(rows):
                Y[row, :] = scipy_signal.filtfilt(b, a, X[row, :])
        if axis == 1:
            for col in range(cols):
                Y[:, col] = scipy_signal.filtfilt(b, a, X[:, col])
        return Y
        
