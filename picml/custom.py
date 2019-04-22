"""
A stash of custom Keras objects and helpers
"""



import keras
import keras.backend as K
import numpy as np
import tensorflow as tf


class CLR(keras.optimizers.Optimizer):
    """
    Supports clipnorm clipvalue
     bounds - a pair of lower and upper learning_rate
     step_function - 'triangular', 'sinusoidal_pulse'
    """

    step_functions = {
        'triangular': lambda i, bounds, steps: K.control_flow_ops.cond(K.equal((i // steps) % 2, 0),
            lambda: (bounds[1] - ((i % steps)) * ((bounds[1] - bounds[0]) // (steps - 1))),
            lambda: (bounds[0] + ((i % steps)) * ((bounds[1] - bounds[0]) // (steps - 1))))
        }

    def __init__(self, bounds=(0.01, 3.), steps=15, momentum_bounds=0., step_function='triangular', **kwargs):
        super(CLR, self).__init__(**kwargs)
        self.learning_rate_bounds = K.constant(bounds)
        self.momentum_bounds = K.constant(momentum_bounds if type(momentum_bounds) in [tuple, list] else (momentum_bounds, momentum_bounds))
        self.steps = K.constant(steps)
        self.step_function = self.step_functions[step_function]
        self.lr = K.variable(bounds[0], name='lr')
        self.momentum = K.variable(momentum_bounds[0] if type(momentum_bounds) in [tuple, list] else momentum_bounds, name='momentum')
        self.iterations = K.variable(0, name='iterations')
        K.get_session().run(self.iterations.initializer)

    def get_updates(self, params, constraints, loss):
        gradients = self.get_gradients(loss, params)
        self.updates = []
        self._update_runtime_parameters()  # this needs to be integrated into the loop...
        shapes = [K.get_variable_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        for param, grad, moment in zip(params, gradients, moments):
            velocity = self.momentum * moment - self.lr * grad
            self.updates.append(K.update(moment, velocity))
            new_param = param + velocity
            if param in constraints.keys():
                constraint = constraints[param]
                new_param = constraint(new_param)
            self.updates.append(K.update(param, new_param))
        return self.updates

    def get_config(self):
        config = {'learning_rate_bounds': self.learning_rate_bounds,
                    'momentum_bounds': self.momentum_bounds,
                    'steps': self.steps,
                    'step_function': self.step_function}
        base_config = super(CLR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _update_runtime_parameters(self):
        self.lr = K.update(self.lr, K.variable(self.step_function(self.iterations, self.learning_rate_bounds, self.steps)))
        self.momentum = K.update(self.momentum, K.variable(self.step_function(self.iterations, self.momentum_bounds, self.steps)))
        self.updates.append(K.update(self.iterations, self.iterations + 1))


class AllPassKernelInitializer(keras.initializers.Initializer):
    """
    Initialize LSTMs to have specific weights on forget gates
    TODO: Shouldn't it have noise?
    """
    def __init__(self, is_gru = False, value = 5.):
        self.is_gru = is_gru
        self.value = value

    def __call__(self, shape, dtype=None):
        if self.is_gru:
            return NotImplemented
        else:
            step = int(shape[0] / 4)
            width = shape[1]
            print(step, width)
            assert step == width
            return K.concatenate([K.zeros((step, width), dtype = dtype),
                K.constant(self.value, shape = K.eye(step), dtype = dtype),
                K.zeros((step, width), dtype = dtype),
                K.zeros((step, width), dtype = dtype)])


class Constr:
    """
    Limit all non-diagonal weights to zero
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.mask = np.repeat(np.eye(X), Y).reshape(X, X, Y).swapaxes(2, 0)
        self.mask = K.constant(self.mask)

    def __call__(self, tensor):
        return tensor * self.mask

    def get_config(self):
        return {"X": self.X, "Y": self.Y}

def maximal_error(x, y):
    """
    Basically an L_infty norm
    """
    return K.max(K.abs(y - x))

def identity_loss(true, pred):
    """
    Helper to input to custom loss networks
    """
    return pred

def l4_loss(true, pred):
    """
    L4 loss - experimental; should create heavy attenuation of outliers
    """
    return K.mean(K.square(K.square(pred - true)), axis=-1)

custom_objects = {"CLR": CLR,
                  "Constr": Constr,
                  "maximal_error": maximal_error,
                  "tf": tf,
                  "identity_loss": identity_loss,
                  "l4_loss": l4_loss}
