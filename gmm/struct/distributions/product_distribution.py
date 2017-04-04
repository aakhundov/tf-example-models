# -*- coding: utf-8 -*-

import tensorflow as tf

from distribution_base import DistributionBase


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
