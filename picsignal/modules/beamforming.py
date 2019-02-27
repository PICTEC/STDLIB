import itertools
import keras
import keras.backend as K
import librosa
import math
import numpy as np
import scipy as sp
import scipy.optimize


def mvdr():
    pass


class DeepMVDR:
    """
    This class wraps the process of beamforming with MVDR and subsystemms requried for MVDR.
    It loads a model from prespecified path, uses the outputs of the model
    to determine dominant source and uses the dominant sources in updating covariance
    matrices of both noise and speech. Covariance matrix of speech is used
    to estimate direction of incidence of sound from the main source.
    """

    def __init__(self, n, frame_len, delay_and_sum, use_channels, model_name, choose=None):
        self.model_path = model_name
        self.mask_thresh_speech = 0.7
        self.mask_thresh_noise = 0.3
        self.num_of_mics = len(choose) if choose else n
        self.delay_and_sum = delay_and_sum # TO DO
        self.use_channels = use_channels  # TO DO
        self.frame_len = frame_len
        self.psd_tracking_constant_speech = 0.95 + 0j
        self.psd_tracking_constant_noise = 0.99 + 0j
        self.choose = choose
        self.frame = 0
        self.fft_len = int(self.frame_len / 2 + 1)
        self.eigenvector = np.ones((self.fft_len, self.num_of_mics), dtype=np.complex64) +\
                           np.zeros((self.fft_len, self.num_of_mics), dtype=np.complex64) * 1j
        self.psd_speech = np.tile(np.diag(np.ones(self.num_of_mics)), (self.fft_len, 1)).reshape(-1, self.num_of_mics, self.num_of_mics).astype(np.complex64)
        self.psd_noise = np.tile(np.diag(np.ones(self.num_of_mics)), (self.fft_len, 1)).reshape(-1, self.num_of_mics, self.num_of_mics).astype(np.complex64)
        self.frequency = 16000
        self.speed_of_sound = 340
        self.doa = np.pi/2
        self.doa_ma = 0.8

    def fast_mvdr(self, sound):
        cminv = np.linalg.inv(self.psd_noise)
        conj = np.conj(self.eigenvector).reshape(self.fft_len, 1, -1)
        return (conj @ cminv @ sound.reshape(self.fft_len, -1, 1)) / (
                conj @ cminv @ self.eigenvector.reshape(self.fft_len, -1, 1))

    def update_psds(self, fft_vector, speech_mask, noise_mask):
        toUpd = speech_mask
        self.psd_speech[toUpd] = self.psd_tracking_constant_speech * self.psd_speech[toUpd] + \
                                 (1 - self.psd_tracking_constant_speech) * \
                                 np.einsum('...i,...j->...ij', fft_vector, fft_vector.conj())[toUpd]
        toUpd = noise_mask
        self.psd_noise[toUpd] = self.psd_tracking_constant_noise * self.psd_noise[toUpd] + \
                                (1 - self.psd_tracking_constant_noise) * \
                                np.einsum('...i,...j->...ij', fft_vector, fft_vector.conj())[toUpd]

    def update_ev_by_power_iteration(self):
        unnormalized_eigenvector = np.einsum('...ij,...j->...i', self.psd_speech, self.eigenvector, dtype=np.complex128)
        eigen_norm = np.sqrt((unnormalized_eigenvector * unnormalized_eigenvector.conj()).mean(1))
        self.eigenvector = unnormalized_eigenvector / eigen_norm[:,None]
        # self.eigenvector2 = np.linalg.eig(self.psd_speech)[0]

    def initialize(self):
        """
        Initialize the model - preload and perform some dry runs o reduce latency
        """
        self.model = keras.models.load_model(self.model_path)
        self.model._make_predict_function()
        self.input = self.model.input
        self.output = self.model.output
        self.session = K.get_session()
        # Three dry run to compile this magical device
        for i in range(3):
            prep = np.random.random([self.num_of_mics, 1, self.fft_len]).astype(np.float32)
            self.session.run(self.output,
                feed_dict={self.input: prep})

    def process(self, ffts):
        """
        Process the sample - accepts single time frame with multiple channels.
        Returns beamformed signal. Uses LSTM masking as a part of beamforming process.
        """
        if self.choose is not None:
            ffts = ffts[:, self.choose]
        prep = ffts.T.reshape(self.num_of_mics, 1, -1)
        prep = np.abs(prep)
        self.doa = self.doa_ma * self.doa + (1 - self.doa_ma) * self.calc_angle(ffts)
        response = self.session.run(self.output,
            feed_dict={self.input: prep})
        vad_mask = np.transpose(response, [2, 0, 1])
        vad_mean =  vad_mask.mean((1,2))
        speech_update = vad_mean > self.mask_thresh_speech
        # print(speech_update.mean())
        noise_update = vad_mean < self.mask_thresh_noise
        # print(noise_update.mean())
        self.update_psds(ffts, speech_update, noise_update)
        self.update_ev_by_power_iteration()
        result_fftd = self.fast_mvdr(vad_mask.reshape(self.fft_len, self.num_of_mics) ** 2 * ffts).astype(np.complex64)
        return result_fftd.reshape(-1, 1)


def gev(spectrogram, signalCovarianceMatrix, noiseCovarianceMatrix):
    GEVinitialization = np.zeros((spectrogram.shape[0], ))

    def GEVmaximizable(w, sigCov, noiseCov):
        return -1 * H(w).dot(sigCov).dot(w) / H(w).dot(noiseCov).dot(w)
    returnedSpec = np.zeros(spectrogram.shape[1:])
    for freq in range(spectrogram.shape[1]):
        weights = np.zeros((spectrogram.shape[0], spectrogram.shape[2]))
        for time in range(spectrogram.shape[2]):
            weights[:, time] = sp.optimize.minimize(GEVmaximizable, GEVinitialization, args=(signalCovarianceMatrix[freq, time, :, :], noiseCovarianceMatrix[freq, time, :, :]))
        returnedSpec[freq, :] = sum(weights * spectrogram[:, freq, :], axis=1)
    returnedSpec
