# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from distribution_base import DistributionBase
from full_covariance import FullCovariance


class GaussianDistribution(DistributionBase):

    def __init__(self, dims, mean=None, covariance=None):
        self.dims = dims
        self.mean = mean
        self.covariance = covariance

        self.tf_mean = None
        self.tf_covariance = None
        self.tf_ln2piD = None

    def initialize(self, dtype=tf.float64):
        if self.tf_mean is None:
            if self.mean is not None:
                self.tf_mean = tf.Variable(self.mean, dtype=dtype)
            else:
                self.tf_mean = tf.Variable(tf.cast(tf.fill([self.dims], 0.0), dtype))

        if self.tf_covariance is None:
            if self.covariance is not None:
                self.tf_covariance = self.covariance
            else:
                self.tf_covariance = FullCovariance(self.dims)

            self.tf_covariance.initialize(dtype)

        if self.tf_ln2piD is None:
            self.tf_ln2piD = tf.constant(np.log(2 * np.pi) * self.dims, dtype=dtype)

    def get_parameters(self):
        return [
            self.tf_mean,
            self.tf_covariance.get_matrix()
        ]

    def get_log_probabilities(self, data):
        tf_quadratic_form = self.tf_covariance.get_inv_quadratic_form(data[0], self.tf_mean)
        tf_log_coefficient = self.tf_ln2piD + self.tf_covariance.get_log_determinant()

        return -0.5 * (tf_log_coefficient + tf_quadratic_form)

    def get_parameter_updaters(self, data, gamma_weighted, gamma_sum):
        tf_new_mean = tf.reduce_sum(data[0] * tf.expand_dims(gamma_weighted, 1), 0)
        tf_covariance_updater = self.tf_covariance.get_value_updater(
            data[0], tf_new_mean, gamma_weighted, gamma_sum)

        return [tf_covariance_updater, self.tf_mean.assign(tf_new_mean)]
