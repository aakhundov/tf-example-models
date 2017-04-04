# -*- coding: utf-8 -*-

import tensorflow as tf

from covariance_base import CovarianceBase


class DiagonalCovariance(CovarianceBase):

    def __init__(self, dims, vector=None, prior=None):
        self.dims = dims
        self.vector = vector
        self.prior = prior
        self.has_prior = None

        self.tf_variance_vector = None
        self.tf_alpha = None
        self.tf_beta = None

    def initialize(self, dtype=tf.float64):
        if self.tf_variance_vector is None:
            if self.vector is not None:
                self.tf_variance_vector = tf.Variable(self.vector, dtype=dtype)
            else:
                self.tf_variance_vector = tf.Variable(tf.cast(tf.fill([self.dims], 1.0), dtype))

        if self.has_prior is None:
            if self.prior is not None:
                self.has_prior = True
                self.tf_alpha = tf.constant(self.prior["alpha"], dtype=dtype)
                self.tf_beta = tf.constant(self.prior["beta"], dtype=dtype)
            else:
                self.has_prior = False

    def get_matrix(self):
        return tf.diag(self.tf_variance_vector)

    def get_inv_quadratic_form(self, data, mean):
        tf_sq_distances = tf.squared_difference(data, tf.expand_dims(mean, 0))

        return tf.reduce_sum(tf_sq_distances / self.tf_variance_vector, 1)

    def get_log_determinant(self):
        return tf.reduce_sum(tf.log(self.tf_variance_vector))

    def get_prior_adjustment(self, original, gamma_sum):
        tf_adjusted = original
        tf_adjusted *= gamma_sum
        tf_adjusted += (2.0 * self.tf_beta)
        tf_adjusted /= gamma_sum + (2.0 * (self.tf_alpha + 1.0))

        return tf_adjusted

    def get_value_updater(self, data, new_mean, gamma_weighted, gamma_sum):
        tf_new_sq_distances = tf.squared_difference(data, tf.expand_dims(new_mean, 0))
        tf_new_variance = tf.reduce_sum(tf_new_sq_distances * tf.expand_dims(gamma_weighted, 1), 0)

        if self.has_prior:
            tf_new_variance = self.get_prior_adjustment(tf_new_variance, gamma_sum)

        return self.tf_variance_vector.assign(tf_new_variance)
