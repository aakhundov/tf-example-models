# -*- coding: utf-8 -*-

import tensorflow as tf

from distribution_base import DistributionBase


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
