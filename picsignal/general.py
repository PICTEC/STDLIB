import numpy as np

H = lambda matrix: np.conj(np.transpose(matrix))
np.H = H  # share namespace (not sure if it works)


def wiener(spec, sigma):
    if type(sigma) in [int, float, complex] or sigma.ndim == 1:
        cleanPSD = (np.sum(np.abs(spec) ** 2, axis=1) / spec.shape[1]) - sigma
    else:
        cleanPSD = np.abs(spec) ** 2 - sigma
    transfer = cleanPSD / (cleanPSD + sigma)
    if transfer.ndim == 1:
        for time in xrange(spec.shape[1]):
            spec[:, time] = transfer * spec[:, time]
    elif transfer.ndim == 2:
        spec = transfer * spec
    return spec

def cqt(x, sr=16000, per_octave = 60, ):
    base_fq = float(sr) / len(x)
    octaves = np.log2((sr/2) / base_fq)
    n_bins = np.floor(octaves * per_octave).astype(np.int32)
    step = 2 ** (1. / per_octave)
    Q = (step ** 0.5) # half semitone(if per_octave = 12) in each direction
    X = np.zeros(n_bins, dtype=np.complex64)
    N = Q * (sr / 2) / (base_fq * step ** np.arange(n_bins))
    for k in range(n_bins):
        X[k] = np.sum(x * np.exp(-2 * 1j * np.pi * Q / N[k] * np.arange(len(x)))) /  N[k]
    return X

def get_n_fft(magnitude): # To hide
    return magnitude.shape[-1] * 2 - 2

def get_stft(n_fft): # To hide
    n_hop = n_fft // 4
    window = np.hanning(n_fft)
    spec_width = n_fft // 2 + 1
    def stft(signal):
        length = 1 + (signal.shape[0] - n_fft) // n_hop
        spec = xp.zeros([length, spec_width], dtype=xp.complex)
        for index in range(length):
            spec[index, :] = np.fft.rfft(window * signal[
                index*n_hop : index*n_hop + n_fft
            ])
        return spec
    return stft

def get_istft(n_fft): # To hide
    n_hop = n_fft // 4
    def istft(spectro):
        length = n_fft + (spectro.shape[0] - 1) * n_hop
        signal = np.zeros(length)
        for index in range(spectro.shape[0]):
            signal[index*n_hop : index*n_hop + n_fft] += np.fft.irfft(spectro[index,:])
        return signal
    return istft

def griffin_lim(magnitude, iterations=250, verbose=False):
    """
    Works on power spectrograms, not log spectrograms!!!
    """
    n_fft = get_n_fft(magnitude)  # check whether static offset is present
    n_hop = n_fft // 4
    signal_length = (magnitude.shape[0] - 1) * n_hop + n_fft
    signal = np.random.random(signal_length)
    stft = get_stft(n_fft)
    istft = get_istft(n_fft)
    for iteration in range(iterations):
        reconstruction = stft(signal)
        phase = np.angle(reconstruction)
        proposed = magnitude * np.exp(1j * phase)
        prev_signal = signal
        signal = istft(proposed)
        RMSE = np.sqrt(((prev_signal - signal)**2).mean())
        if verbose:
            print("Iteration {}/{} RMSE: {}".format(iteration + 1, iterations, RMSE))
    return signal


