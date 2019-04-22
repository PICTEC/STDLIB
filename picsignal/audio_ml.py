import keras.backend as K
import tensorflow as tf

try:
    import tensorflow.signal as tfsignal
    import tensorflow.spectral as tfspectral
except ImportError:
    import tensorflow.contrib.signal as tfsignal
    import tensorflow.spectral as tfspectral


class LossSum:
    def __init__(self, *components):
        self.components = components

    def __call__(self, true, preds):
        return sum([comp(true, preds) for comp in self.components])


class MFCCLossComponent:
    def __init__(self, rate=0.1, depth=26, n_fft=257, sample_rate=16000, bounds=(50, 8000)):
        self.depth = depth
        self.rate = K.cast_to_floatx(rate)
        self.weights = tfsignal.linear_to_mel_weight_matrix(
            num_mel_bins=depth,
            num_spectrogram_bins=n_fft,
            sample_rate=sample_rate,
            lower_edge_hertz=bounds[0],
            upper_edge_hertz=bounds[1]
        )

    def __call__(self, true, preds):
        mfcc_true = tfspectral.dct(K.dot(true, self.weights))
        mfcc_preds = tfspectral.dct(K.dot(preds, self.weights))
        return self.rate * K.mean((mfcc_true - mfcc_preds)**2)


def pesq(gt, pred, phase):
    """
    Calls `PESQ +wb +16000` on a recordings, the recording should be in 16kHz and PESQ
    should be installed. It does fork(), take care of memory
    """
    spec = (np.sqrt(np.exp(-gt)) * 512) * np.exp(phase * 1j)
    sound = np.zeros(spec.shape[0] * 128 + 512 - 128)
    for i in range(spec.shape[0]):
        frame = np.fft.irfft(spec[i,:])
        sound[128 * i : 128 * i + 512] += frame
    spec = (np.sqrt(np.exp(-pred)) * 512) * np.exp(phase * 1j)
    sound2 = np.zeros(spec.shape[0] * 128 + 512 - 128)
    for i in range(spec.shape[0]):
        frame = np.fft.irfft(spec[i,:])
        sound2[128 * i : 128 * i + 512] += frame
    fname_gt = tempfile.mktemp() + ".wav"
    fname_pred = tempfile.mktemp() + ".wav"
    # print(sound.shape, sound2.shape)
    sio.write(fname_gt, 16000, (2**15 * sound).astype(np.int16))
    sio.write(fname_pred, 16000, (2**15 * sound2).astype(np.int16))
    ot, e = subprocess.Popen(["PESQ", "+wb", "+16000", fname_gt, fname_pred], stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()
    os.remove(fname_gt)
    os.remove(fname_pred)
    # print(ot)
    o = ot.decode("utf-8").split('\n')[-2]
    # print(o, len(o))
    # if not len(o):
    #     print(ot.decode("utf-8"))
    value = re.findall("= \d\.\d+", o)[0]
    # print(value)
    return float(value[2:])

def LSD(gt, pred):
    """
    Log-spectral distance
    """
    innermost = (10 * ((-pred) - (-gt)) / np.log(10)) ** 2
    for i in range(gt.shape[0]):
        inner = innermost[i, :, :]
        length = len(np.where((gt[i, :, :] != 0).sum(1))[0])
        inner = inner[:length]
        sublsd = []
        for t in range(length):
            step = 2 / 513
            frame = inner[t]
            integral = frame.sum()
            sublsd.append(np.sqrt(step * integral))
    return np.array(sublsd)
