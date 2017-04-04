# -*- coding: utf-8 -*-

import tensorflow as tf

from covariance_base import CovarianceBase


class FullCovariance(CovarianceBase):

    def __init__(self, dims, matrix=None, prior=None):
        self.dims = dims
        self.matrix = matrix
        self.prior = prior
        self.has_prior = None

        self.tf_covariance_matrix = None
        self.tf_alpha = None
        self.tf_beta = None

    def initialize(self, dtype=tf.float64):
        if self.tf_covariance_matrix is None:
            if self.matrix is not None:
                self.tf_covariance_matrix = tf.Variable(self.matrix, dtype=dtype)
            else:
                self.tf_covariance_matrix = tf.Variable(tf.diag(tf.cast(tf.fill([self.dims], 1.0), dtype)))

        if self.has_prior is None:
            if self.prior is not None:
                self.has_prior = True
                self.tf_alpha = tf.constant(self.prior["alpha"], dtype=dtype)
                self.tf_beta = tf.constant(self.prior["beta"], dtype=dtype)
            else:
                self.has_prior = False

    def get_matrix(self):
        return self.tf_covariance_matrix

    def get_inv_quadratic_form(self, data, mean):
        tf_differences = tf.subtract(data, tf.expand_dims(mean, 0))
        tf_diff_times_inv_cov = tf.matmul(tf_differences, tf.matrix_inverse(self.tf_covariance_matrix))

        return tf.reduce_sum(tf_diff_times_inv_cov * tf_differences, 1)

    def get_log_determinant(self):
        tf_eigvals = tf.self_adjoint_eigvals(self.tf_covariance_matrix)

        return tf.reduce_sum(tf.log(tf_eigvals))

    def get_prior_adjustment(self, original, gamma_sum):
        tf_adjusted = original
        tf_adjusted *= gamma_sum
        tf_adjusted += tf.diag(tf.fill([self.dims], 2.0 * self.tf_beta))
        tf_adjusted /= gamma_sum + (2.0 * (self.tf_alpha + 1.0))

        return tf_adjusted

    def get_value_updater(self, data, new_mean, gamma_weighted, gamma_sum):
        tf_new_differences = tf.subtract(data, tf.expand_dims(new_mean, 0))
        tf_sq_dist_matrix = tf.matmul(tf.expand_dims(tf_new_differences, 2), tf.expand_dims(tf_new_differences, 1))
        tf_new_covariance = tf.reduce_sum(tf_sq_dist_matrix * tf.expand_dims(tf.expand_dims(gamma_weighted, 1), 2), 0)

        if self.has_prior:
            tf_new_covariance = self.get_prior_adjustment(tf_new_covariance, gamma_sum)

        return self.tf_covariance_matrix.assign(tf_new_covariance)
