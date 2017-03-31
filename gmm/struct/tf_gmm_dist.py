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

        self._mean = None
        self._covariance = None
        self._ln2piD = None

    def initialize(self, dtype=tf.float64):
        if self._mean is None:
            if self.mean is not None:
                self._mean = tf.Variable(self.mean, dtype=dtype)
            else:
                self._mean = tf.Variable(tf.cast(tf.fill([self.dims], 0.0), dtype))

        if self._covariance is None:
            if self.covariance is not None:
                self._covariance = self.covariance
            else:
                self._covariance = tf_gmm_cov.DiagonalCovariance(self.dims)

            self._covariance.initialize(dtype)

        if self._ln2piD is None:
            self._ln2piD = tf.constant(np.log(2 * np.pi) * self.dims, dtype=dtype)

    def get_parameters(self):
        return [
            self._mean,
            self._covariance.get_matrix()
        ]

    def get_log_probabilities(self, data):
        quadratic_form = self._covariance.get_quadratic_form(data[0], self._mean)
        log_coefficient = self._ln2piD + self._covariance.get_log_determinant()

        return -0.5 * (log_coefficient + quadratic_form)

    def get_parameter_updaters(self, data, gamma_weighted, gamma_sum):
        new_mean = tf.reduce_sum(data[0] * tf.expand_dims(gamma_weighted, 1), 0)
        covariance_updater = self._covariance.get_value_updater(
            data[0], new_mean, gamma_weighted, gamma_sum)

        return [covariance_updater, self._mean.assign(new_mean)]


class CategoricalDistribution(DistributionBase):

    def __init__(self, counts, means=None):
        self.dims = len(counts)
        self.counts = counts
        self.means = means

        self._means = None

    def initialize(self, dtype=tf.float64):
        if self._means is None:
            self._means = []
            for dim in range(self.dims):
                if self.means is not None:
                    mean = tf.Variable(self.means[dim], dtype=dtype)
                else:
                    rand = tf.random_uniform([self.counts[dim]], maxval=1.0, dtype=dtype)
                    mean = tf.Variable(rand / tf.reduce_sum(rand))
                self._means.append(mean)

    def get_parameters(self):
        return self._means

    def get_log_probabilities(self, data):
        log_probabilities = []
        for dim in range(self.dims):
            log_means = tf.log(self._means[dim])
            log_probabilities.append(
                tf.gather(log_means, data[0][:, dim])
            )

        stacked = tf.parallel_stack(log_probabilities)

        return tf.reduce_sum(stacked, axis=0)

    def get_parameter_updaters(self, data, gamma_weighted, gamma_sum):
        parameter_updaters = []
        for dim in range(self.dims):
            partition = tf.dynamic_partition(gamma_weighted, data[0][:, dim], self.counts[dim])
            new_means = tf.parallel_stack([tf.reduce_sum(p) for p in partition])
            parameter_updaters.append(self._means[dim].assign(new_means))

        return parameter_updaters


class ProductDistribution(DistributionBase):

    def __init__(self, distributions):
        self.count = len(distributions)
        self.distributions = distributions

    def initialize(self, dtype=tf.float64):
        for dist in self.distributions:
            dist.initialize(dtype)

    def get_parameters(self):
        return [dist.get_parameters() for dist in self.distributions]

    def get_log_probabilities(self, data):
        log_probabilities = []
        for dist in range(self.count):
            log_probabilities.append(
                self.distributions[dist].get_log_probabilities(
                    [data[dist]]
                )
            )

        stacked = tf.parallel_stack(log_probabilities)

        return tf.reduce_sum(stacked, axis=0)

    def get_parameter_updaters(self, data, gamma_weighted, gamma_sum):
        parameter_updaters = []
        for dist in range(self.count):
            parameter_updaters.extend(
                self.distributions[dist].get_parameter_updaters(
                    [data[dist]], gamma_weighted, gamma_sum
                )
            )

        return parameter_updaters
