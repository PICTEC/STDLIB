import argparse
import keras
import keras.backend as K
from keras.layers import Lambda, LeakyReLU, Conv2D, Flatten, TimeDistributed, Dense, Input
from keras.models import load_model, Model
from keras.optimizers import Adam
from keras.regularizers import L1L2
import logging
import math
import numpy as np
import os
import scipy as sp
import scipy.optimize
import tensorflow as tf
import time
from picml.utils import BufferMixin, StopOnConvergence, save_model, list_sounds, open_sound, stft

import numpy as np

import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Lambda, LeakyReLU, TimeDistributed, Flatten, BatchNormalization, concatenate
import keras.backend as K
import keras.callbacks as kc


logger = logging.getLogger("picsignal.modules")


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


def default_model(n_fft):
    """
    This model is a bit too large for Tegra, but it is proven
    """
    assert n_fft == 257, "Default model cannot handle non-257 fft sizes"
    input_lower = Input((None, 257), name="input_lf")
    layer = Lambda(K.expand_dims)(input_lower)
    layer = LeakyReLU(0.01)(Conv2D(12, kernel_size=(9, 1), activation='linear')(layer))
    layer = LeakyReLU(0.01)(Conv2D(12, kernel_size=(1, 5), activation='linear')(layer))
    layer = LeakyReLU(0.01)(Conv2D(12, kernel_size=(9, 1), activation='linear')(layer))
    layer = LeakyReLU(0.01)(Conv2D(12, kernel_size=(1, 5), activation='linear')(layer))
    layer = TimeDistributed(Flatten())(layer)
    layer = LeakyReLU(0.01)(Dense(1024, kernel_regularizer=L1L2(l1=1e-5))(layer))
    layer = LeakyReLU(0.01, name='hidden')(Dense(512, kernel_regularizer=L1L2(l1=1e-5))(layer))
    layer = LeakyReLU(0.01)(Dense(350, kernel_regularizer=L1L2(l2=1e-5))(layer))
    layer = Dense(257)(layer)
    mdl = Model(input_lower, layer)
    mdl.summary()
    return mdl


def fast_model(n_fft):
    """
    Architecture of a faster model
    """
    input_lower = Input((None, n_fft), name="input_lf")
    layer = Lambda(K.expand_dims)(input_lower)
    layer = LeakyReLU(0.01)(Conv2D(12, kernel_size=(17, 1), activation='linear')(layer))
    layer = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 3, 2)))(layer)
    layer = LeakyReLU(0.01)(Dense(n_fft // 2)(layer))
    layer = LeakyReLU(0.01)(Dense(n_fft // 4)(layer))
    layer = TimeDistributed(Flatten())(layer)
    layer = LeakyReLU(0.01)(Dense(2 * n_fft, kernel_regularizer=L1L2(l1=1e-5))(layer))
    layer = LeakyReLU(0.01)(Dense(n_fft, kernel_regularizer=L1L2(l1=1e-5))(layer))
    layer = LeakyReLU(0.01, name='hidden')(Dense(3 * n_fft // 4, kernel_regularizer=L1L2(l1=1e-5))(layer))
    layer = Dense(n_fft)(layer)
    mdl = Model(input_lower, layer)
    mdl.summary()
    return mdl

def faster_model(n_fft):
    """
    Architecture of a faster model
    """
    input_lower = Input((None, n_fft), name="input_lf")
    layer = LeakyReLU(0.01)(Dense(n_fft, kernel_regularizer=L1L2(l1=1e-5))(input_lower))
    layer = LeakyReLU(0.01)(Dense(2 * n_fft // 3, kernel_regularizer=L1L2(l1=1e-5))(layer))
    layer = LeakyReLU(0.01)(Dense(n_fft // 2, kernel_regularizer=L1L2(l1=1e-5))(layer))
    layer = Lambda(lambda x: K.expand_dims(x))(layer)
    layer = LeakyReLU(0.01)(Conv2D(12, kernel_size=(17, 1), activation='linear')(layer))
    layer = TimeDistributed(Flatten())(layer)
    layer = LeakyReLU(0.01, name='hidden')(Dense(3 * n_fft // 4, kernel_regularizer=L1L2(l1=1e-5))(layer))
    layer = Dense(n_fft)(layer)
    mdl = Model(input_lower, layer)
    mdl.summary()
    return mdl



class DAEPostFilter(BufferMixin([17, 257, 1], np.complex64)):
    """
    Postfilter for signal based on DAE. The DAE accepts a context of time, therefore
    class inherits BufferMixin. Class contains methods to train the postfilter.
    """

    _all_imports = {}
    _all_imports.update(dae.imports)
    _models = {"default": default_model,
               "fast": fast_model,
               "faster": faster_model}

    def __init__(self, fname="storage/dae-pf.h5", n_fft=1024):
        super().__init__()
        self.model = load_model(fname, self._all_imports)
        self.model._make_predict_function()
        self.input = self.model.input
        self.output = self.model.output
        self.session = K.get_session()
        fft_size = n_fft // 2 + 1
        assert self.model.input_shape[-1] == fft_size, "Input shape is {}; model requires {}".format(fft_size, self.model.input_shape[-1])
        assert self.model.output_shape[-1] == fft_size, "Input shape is {}; model requires {}".format(fft_size, self.model.output_shape[-1])

    def initialize(self):
        """
        Call this before processing starts
        """

    def process(self, sample):
        """
        Accept single mono sample, push to the rolling buffer and then process the buffer
        with the model. Input and output of model is log-power. Phase is reapplied at
        the end of processing,
        """
        self.buffer.push(sample)
        predictive = -np.log(np.abs(self.buffer.reshape([1, 17, 257])) ** 2 + 2e-12)
        result = self.session.run(self.output,
                    feed_dict={self.input: predictive})
        result = result[:1, 0, :].T  # extract channel of interest
        result = np.sqrt(np.exp(-result)) * np.exp(1j * np.angle(sample))  # restore phase information
        return result

    @classmethod
    def train(cls, model_config, train_X, train_Y, valid_ratio=0.1, path_to_save="storage/dae-pf.h5", n_fft=512):
        """
        This should create a model from some training script...
        train_X should be padded by 16 from the beginning of the recording...
        n_fft - determines the size of the network
        """
        fft_size = n_fft // 2 + 1
        spec = cls._models[model_config] if isinstance(model_config, str) else model_config
        model = spec(fft_size)
        sel = np.random.random(len(train_X)) > valid_ratio
        train_X, valid_X = train_X[sel], train_X[~sel]
        train_Y, valid_Y = train_Y[sel], train_Y[~sel]
        for lr in [0.0003, 0.0001, 0.00003]:
            model.compile(optimizer=Adam(lr, clipnorm=1.), loss='mse')
            model.fit(train_X, train_Y, validation_data=[valid_X, valid_Y], epochs=50,
                        callbacks=[StopOnConvergence(5)], batch_size=8)
        save_model(model, path_to_save)
        return model

    @staticmethod
    def test(model, test_X, test_Y):
        pass


def list_dataset(clean, noisy):
    cleans = set([x.split(os.sep)[-1] for x in list_sounds(clean)])
    noises = set([x.split(os.sep)[-1] for x in list_sounds(noisy)])
    fnames = cleans | noises
    return [os.path.join(clean, x) for x in fnames], [os.path.join(noisy, x) for x in fnames]

def get_dataset(clean, noisy, ratio=0.2, maxlen=1200, n_fft=512):
    fft_size = n_fft // 2 + 1
    clean, noisy = list_dataset(clean, noisy)
    assert clean, "No data with common filenames"
    assert noisy, "No data with common filenames"
    X = np.zeros([len(clean), maxlen + 16, fft_size], np.float32)
    Y = np.zeros([len(clean), maxlen, fft_size], np.float32)
    sel = np.random.random(len(clean)) > ratio
    for ix, (cl, ns) in enumerate(zip(clean, noisy)):
        print("Loading file", ix)
        cl, ns = open_sound(cl), open_sound(ns)
        assert cl[0] == ns[0]
        cl, ns = cl[1], ns[1]
        if len(ns.shape) > 1:
            ns = ns[:, 0]
        spec = -np.log(np.abs(stft(cl, n_fft=n_fft)) ** 2 + 2e-12).T[:maxlen]
        spec = np.pad(spec, ((16, maxlen - spec.shape[0]), (0, 0)), 'constant', constant_values=-np.log(2e-12))
        X[ix, :, :] = spec
        spec = -np.log(np.abs(stft(ns, n_fft=n_fft)) ** 2 + 2e-12).T[:maxlen]
        spec = np.pad(spec, ((0, maxlen - spec.shape[0]), (0, 0)), 'constant', constant_values=-np.log(2e-12))
        Y[ix, :, :] = spec
    return [X[sel], Y[sel]], [X[~sel], Y[~sel]]


class MonoModel:
    """
    MonoModel wraps monophonic masking into a simple model usage within Runtime.
    This models loads its' neural network from `path` and prepares it for fast
    evaluation. Model is assumed to accept plain absolute values of spectrum and
    return a soft mask for that spectrum. Masks are scaled and clipped before
    application to the data.
    """
    
    def __init__(self, path, scaling_factor=1, clip=0):
        logger.info("Loading TF model")
        self.clip = clip
        self.model = keras.models.load_model(path)
        logger.info("TF model loaded")
        self.scaling_factor = float(scaling_factor)

    def initialize(self):
        """
        Prepare the model - load all required data.
        """
        self.model.reset_states()
        self.model._make_predict_function()
        self.input = self.model.input
        self.output = self.model.output
        self.session = K.get_session()

    def process(self, sample):
        """
        Accept a single multichannel frame. Discard all but one channel
        and perform masking on that channel.
        """
        prep = sample[:, 0].reshape(1, 1, -1)
        prep = np.abs(prep)
        response = self.session.run(self.output,
            feed_dict={self.input: prep})
        response = np.clip(response, 0, None) ** self.scaling_factor
        response[response < self.clip] = 0
        response = response[0, 0, :] * sample[:, 0]
        return response.reshape(-1, 1)


def mk_dense(variant):
    assert variant in ['trim', 'full_separate', 'full_whole']
    def mk_model(**kwargs):
        wrap = lambda x: LeakyReLU(0.01)(x)
        input_lower = Input((None, 129), name="input_lf")
        layer = Lambda(lambda x: x / kwargs["MAX"])(input_lower)
        lyr = Lambda(K.expand_dims)(layer)
        windowing = Conv2D(15, (15, 1), padding='same', use_bias=False)
        lyr = windowing(lyr)
        windowing.set_weights([np.eye(15).reshape(15, 1, 1, 15)])
        windowing.trainable = False
        base = lyr = TimeDistributed(Flatten())(lyr)
        if variant == "trim":
            lyr = wrap(Dense(2048)(lyr))
            lyr = wrap(Dense(2048)(lyr))
            lf_and_hf = Dense(129)(lyr)
        if variant == "full_separate":
            lyr = wrap(Dense(1024)(base))
            lyr = wrap(Dense(1024)(lyr))
            out1 = Dense(129)(lyr)
            lyr = wrap(Dense(1024)(base))
            lyr = wrap(Dense(1024)(lyr))
            out2 = Dense(128)(lyr)
            lf_and_hf = concatenate([out1, out2])
        if variant == "full_whole":
            lyr = wrap(Dense(2048)(lyr))
            lyr = wrap(Dense(2048)(lyr))
            lf_and_hf = Dense(257)(lyr)
        lf_and_hf = Lambda(lambda x: x * kwargs["MAX"])(lf_and_hf)
        mdl = Model(input_lower, lf_and_hf)
        mdl.summary()
        mdl.compile('adam', 'mse')
        return mdl
    return mk_model

def mk_conv_dae(variant, complexity):
    assert variant in ['trim', 'full_separate', 'full_whole']
    def mk_model(**kwargs):
        input_lower = Input((None, 129), name="input_lf")
        layer = Lambda(lambda x: x / kwargs["MAX"])(input_lower)
        base = Lambda(K.expand_dims)(layer)
        if variant == "trim":
            layer = base
            for i in range(complexity):
                layer = LeakyReLU(0.01)(Conv2D(i * 8 + 20, kernel_size=(1,5), padding='same', activation='linear')(layer))
                layer = LeakyReLU(0.01)(Conv2D(i * 8 + 24, kernel_size=(9,1), padding='same', activation='linear')(layer))
            layer = LeakyReLU(0.01)(Conv2D(1, kernel_size=(1,1), padding='same', activation='linear')(layer))
            for i in range(complexity - 1, -1, -1):
                layer = LeakyReLU(0.01)(Conv2D(i * 8 + 24, kernel_size=(9,1), padding='same', activation='linear')(layer))
                layer = LeakyReLU(0.01)(Conv2D(i * 8 + 20, kernel_size=(1,5), padding='same', activation='linear')(layer))     
            lf_and_hf = Dense(129)(TimeDistributed(Flatten())(layer))
        if variant == "full_separate":
            outs = []
            for sz in [129, 128]:
                layer = base
                for i in range(complexity):
                    layer = LeakyReLU(0.01)(Conv2D(i * 4 + 10, kernel_size=(1,5), padding='same', activation='linear')(layer))
                    layer = LeakyReLU(0.01)(Conv2D(i * 4 + 12, kernel_size=(9,1), padding='same', activation='linear')(layer))
                layer = LeakyReLU(0.01)(Conv2D(1, kernel_size=(1,1), padding='same', activation='linear')(layer))
                for i in range(complexity - 1, -1, -1):
                    layer = LeakyReLU(0.01)(Conv2D(i * 4 + 12, kernel_size=(9,1), padding='same', activation='linear')(layer))
                    layer = LeakyReLU(0.01)(Conv2D(i * 4 + 10, kernel_size=(1,5), padding='same', activation='linear')(layer))
                outs.append(Dense(sz)(TimeDistributed(Flatten())(layer)))
            lf_and_hf = concatenate(outs)
        if variant == "full_whole":
            layer = base
            for i in range(complexity):
                layer = LeakyReLU(0.01)(Conv2D(i * 4 + 10, kernel_size=(1,5), padding='same', activation='linear')(layer))
                layer = LeakyReLU(0.01)(Conv2D(i * 4 + 12, kernel_size=(9,1), padding='same', activation='linear')(layer))
            layer = LeakyReLU(0.01)(Conv2D(1, kernel_size=(1,1), padding='same', activation='linear')(layer))
            for i in range(complexity - 1, -1, -1):
                layer = LeakyReLU(0.01)(Conv2D(i * 8 + 24, kernel_size=(9,1), padding='same', activation='linear')(layer))
                layer = LeakyReLU(0.01)(Conv2D(i * 8 + 20, kernel_size=(1,5), padding='same', activation='linear')(layer))
            lf_and_hf = Dense(257)(TimeDistributed(Flatten())(layer))            
        lf_and_hf = Lambda(lambda x: x * kwargs["MAX"])(lf_and_hf)
        mdl = Model(input_lower, lf_and_hf)
        mdl.summary()
        mdl.compile('adam', 'mse')
        return mdl
    return mk_model

def mk_conv_dense(variant, complexity):
    assert variant in ['trim', 'full_separate', 'full_whole']
    def mk_model(**kwargs):
        input_lower = Input((None, 129), name="input_lf")
        layer = Lambda(lambda x: x / kwargs["MAX"])(input_lower)
        base = Lambda(K.expand_dims)(layer)
        if variant == "trim":
            layer = base
            for i in range(complexity):
                layer = LeakyReLU(0.01)(Conv2D(i * 8 + 20, kernel_size=(1,5), padding='same', activation='linear')(layer))
                layer = LeakyReLU(0.01)(Conv2D(i * 8 + 24, kernel_size=(9,1), padding='same', activation='linear')(layer))
            layer = TimeDistributed(Flatten())(layer)
            layer = LeakyReLU(0.01)(Dense(1024)(layer))
            hidden = layer = LeakyReLU(0.01, name='hidden')(Dense(96)(layer))
            for i in range(complexity - 1, -1, -1):
                layer = LeakyReLU(0.01)(Dense(512)(layer))
            lf_and_hf = Dense(129)(layer)
        if variant == "full_separate":
            outs = []
            for sz in [129, 128]:
                layer = base
                for i in range(complexity):
                    layer = LeakyReLU(0.01)(Conv2D(i * 4 + 10, kernel_size=(1,5), padding='same', activation='linear')(layer))
                    layer = LeakyReLU(0.01)(Conv2D(i * 4 + 12, kernel_size=(9,1), padding='same', activation='linear')(layer))
                layer = TimeDistributed(Flatten())(layer)
                layer = LeakyReLU(0.01)(Dense(512)(layer))
                layer = LeakyReLU(0.01)(Dense(96)(layer))
                for i in range(complexity - 1, -1, -1):
                    layer = LeakyReLU(0.01)(Dense(512)(layer))
                outs.append(Dense(sz)(layer))
            lf_and_hf = concatenate(outs)
        if variant == "full_whole":
            layer = base
            for i in range(complexity):
                layer = LeakyReLU(0.01)(Conv2D(i * 4 + 10, kernel_size=(1,5), padding='same', activation='linear')(layer))
                layer = LeakyReLU(0.01)(Conv2D(i * 4 + 12, kernel_size=(9,1), padding='same', activation='linear')(layer))
            layer = TimeDistributed(Flatten())(layer)
            layer = LeakyReLU(0.01)(Dense(1024)(layer))
            hidden = layer = LeakyReLU(0.01, name='hidden')(Dense(96)(layer))
            for i in range(complexity - 1, -1, -1):
                layer = LeakyReLU(0.01)(Dense(512)(layer))
            lf_and_hf = Dense(257)(layer)
        lf_and_hf = Lambda(lambda x: x * kwargs["MAX"])(lf_and_hf)
        mdl = Model(input_lower, lf_and_hf)
        mdl.summary()
        mdl.compile('adam', 'mse')
        return mdl
    return mk_model