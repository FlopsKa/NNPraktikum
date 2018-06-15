# -*- coding: utf-8 -*-

"""
Activation functions which can be used within neurons.
"""

import numpy as np
from numpy import exp
from numpy import divide
from numpy import ones
from numpy import asarray


class Activation(object):
    """
    Containing various activation functions and their derivatives
    """

    @staticmethod
    def sign(net_output, threshold=0):
        return net_output >= threshold

    @staticmethod
    def sigmoid(net_output):
        # use e^x from numpy to avoid overflow
        return 1/(1+exp(-1.0*net_output))

    @staticmethod
    def sigmoid_prime(net_output):
        # Here you have to code the derivative of sigmoid function
        # netOutput.*(1-netOutput)
        return net_output * (1.0 - net_output)

    @staticmethod
    def tanh(net_output):
        # return 2*Activation.sigmoid(2*netOutput)-1
        ex = exp(1.0*net_output)
        exn = exp(-1.0*net_output)
        return divide(ex-exn, ex+exn)  # element-wise division

    @staticmethod
    def tanh_prime(net_output):
        # Here you have to code the derivative of tanh function
        return 1-Activation.tanh(net_output)**2

    @staticmethod
    def rectified(net_output):
        return asarray([max(0.0, i) for i in net_output])

    @staticmethod
    def rectified_prime(net_output):
        # reluPrime=1 if netOutput > 0 otherwise 0
        #print(type(netOutput))
        return net_output > 0

    @staticmethod
    def identity(net_output):
        return net_output

    @staticmethod
    def identity_prime(net_output):
        # identityPrime = 1
        return ones(net_output.size)

    @staticmethod
    def softmax(net_output):
        return exp(net_output) / np.sum(exp(net_output))

    @staticmethod
    def softmax_prime(net_output):
        # case i != j is missing?!
        return net_output * (1.0 - net_output)

    @staticmethod
    def get_activation(activation_func):
        """
        Returns the activation function corresponding to the given string
        """

        if activation_func == 'sigmoid':
            return Activation.sigmoid
        elif activation_func == 'softmax':
            return Activation.softmax
        elif activation_func == 'tanh':
            return Activation.tanh
        elif activation_func == 'relu':
            return Activation.rectified
        elif activation_func == 'linear':
            return Activation.identity
        else:
            raise ValueError('Unknown activation function: ' + activation_func)

    @staticmethod
    def get_derivative(activation_func):
        """
        Returns the derivative function corresponding to a given string which
        specify the activation function
        """

        if activation_func == 'sigmoid':
            return Activation.sigmoid_prime
        elif activation_func == 'softmax':
            return Activation.softmax_prime
        elif activation_func == 'tanh':
            return Activation.tanh_prime
        elif activation_func == 'relu':
            return Activation.rectified_prime
        elif activation_func == 'linear':
            return Activation.identity_prime
        else:
            raise ValueError('Cannot get the derivative of'
                             ' the activation function: ' + activation_func)
