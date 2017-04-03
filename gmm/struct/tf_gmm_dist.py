import numpy as np
import tensorflow as tf

import tf_gmm_cov


class DistributionBase:

    def initialize(self, dtype):
        raise NotImplementedError()

    def get_parameters(self):
        raise NotImplementedError()

    def get_log_probabilities(self, data):
        raise NotImplementedError()

    def get_parameter_updaters(self, data, gamma_weighted, gamma_sum):
        raise NotImplementedError()


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
                self.tf_covariance = tf_gmm_cov.FullCovariance(self.dims)

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


class CategoricalDistribution(DistributionBase):

    def __init__(self, counts, means=None):
        self.dims = len(counts)
        self.counts = counts
        self.means = means

        self.tf_means = None

    def initialize(self, dtype=tf.float64):
        if self.tf_means is None:
            self.tf_means = []
            for dim in range(self.dims):
                if self.means is not None:
                    tf_mean = tf.Variable(self.means[dim], dtype=dtype)
                else:
                    tf_rand = tf.random_uniform([self.counts[dim]], maxval=1.0, dtype=dtype)
                    tf_mean = tf.Variable(tf_rand / tf.reduce_sum(tf_rand))
                self.tf_means.append(tf_mean)

    def get_parameters(self):
        return self.tf_means

    def get_log_probabilities(self, data):
        tf_log_probabilities = []
        for dim in range(self.dims):
            tf_log_means = tf.log(self.tf_means[dim])
            tf_log_probabilities.append(
                tf.gather(tf_log_means, data[0][:, dim])
            )

        return tf.reduce_sum(tf.parallel_stack(tf_log_probabilities), axis=0)

    def get_parameter_updaters(self, data, gamma_weighted, gamma_sum):
        tf_parameter_updaters = []
        for dim in range(self.dims):
            tf_partition = tf.dynamic_partition(gamma_weighted, data[0][:, dim], self.counts[dim])
            tf_new_means = tf.parallel_stack([tf.reduce_sum(p) for p in tf_partition])
            tf_parameter_updaters.append(self.tf_means[dim].assign(tf_new_means))

        return tf_parameter_updaters


class ProductDistribution(DistributionBase):

    def __init__(self, factors):
        self.factors = factors

    def initialize(self, dtype=tf.float64):
        for dist in self.factors:
            dist.initialize(dtype)

    def get_parameters(self):
        return [dist.get_parameters() for dist in self.factors]

    def get_log_probabilities(self, data):
        tf_log_probabilities = []
        for dist in range(len(self.factors)):
            tf_log_probabilities.append(
                self.factors[dist].get_log_probabilities(
                    [data[dist]]
                )
            )

        return tf.reduce_sum(tf.parallel_stack(tf_log_probabilities), axis=0)

    def get_parameter_updaters(self, data, gamma_weighted, gamma_sum):
        tf_parameter_updaters = []
        for dist in range(len(self.factors)):
            tf_parameter_updaters.extend(
                self.factors[dist].get_parameter_updaters(
                    [data[dist]], gamma_weighted, gamma_sum
                )
            )

        return tf_parameter_updaters
