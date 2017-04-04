# -*- coding: utf-8 -*-

import tensorflow as tf

from covariance_base import CovarianceBase


class IsotropicCovariance(CovarianceBase):

    def __init__(self, dims, scalar=None, prior=None):
        self.dims = dims
        self.scalar = scalar
        self.prior = prior
        self.has_prior = None

        self.tf_variance_scalar = None
        self.tf_alpha = None
        self.tf_beta = None
        self.tf_dims = None

    def initialize(self, dtype=tf.float64):
        if self.tf_variance_scalar is None:
            if self.scalar is not None:
                self.tf_variance_scalar = tf.Variable(self.scalar, dtype=dtype)
            else:
                self.tf_variance_scalar = tf.Variable(1.0, dtype=dtype)

        if self.has_prior is None:
            if self.prior is not None:
                self.has_prior = True
                self.tf_alpha = tf.constant(self.prior["alpha"], dtype=dtype)
                self.tf_beta = tf.constant(self.prior["beta"], dtype=dtype)
            else:
                self.has_prior = False

        self.tf_dims = tf.constant(self.dims, dtype=dtype)

    def get_matrix(self):
        return tf.diag(tf.fill([self.dims], self.tf_variance_scalar))

    def get_inv_quadratic_form(self, data, mean):
        tf_sq_distances = tf.squared_difference(data, tf.expand_dims(mean, 0))
        tf_sum_sq_distances = tf.reduce_sum(tf_sq_distances, 1)

        return tf_sum_sq_distances / self.tf_variance_scalar

    def get_log_determinant(self):
        return self.tf_dims * tf.log(self.tf_variance_scalar)

    def get_prior_adjustment(self, original, gamma_sum):
        tf_adjusted = original
        tf_adjusted *= gamma_sum
        tf_adjusted += (2.0 * self.tf_beta)
        tf_adjusted /= gamma_sum + (2.0 * (self.tf_alpha + 1.0))

        return tf_adjusted

    def get_value_updater(self, data, new_mean, gamma_weighted, gamma_sum):
        tf_new_sq_distances = tf.squared_difference(data, tf.expand_dims(new_mean, 0))
        tf_new_variance = tf.reduce_sum(tf_new_sq_distances * tf.expand_dims(gamma_weighted, 1)) / self.tf_dims

        if self.has_prior:
            tf_new_variance = self.get_prior_adjustment(tf_new_variance, gamma_sum)

        return self.tf_variance_scalar.assign(tf_new_variance)
