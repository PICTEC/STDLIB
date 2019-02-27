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
