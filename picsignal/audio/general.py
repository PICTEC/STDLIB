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
