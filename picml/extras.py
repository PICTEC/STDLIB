import keras

import keras.backend as K

class AllPassKernelInitializer(keras.initializers.Initializer):
    def __init__(self, is_gru = False, value = 5.):
        self.is_gru = is_gru
        self.value = value

    def __call__(self, shape, dtype=None):
        if self.is_gru:
            return NotImplemented
        else:
            step = int(shape[0] / 4)
            width = shape[1]
            print step, width
            assert step == width
            return K.concatenate([K.zeros((step, width), dtype = dtype),
                K.constant(self.value, shape = K.eye(step), dtype = dtype),
                K.zeros((step, width), dtype = dtype),
                K.zeros((step, width), dtype = dtype)])
