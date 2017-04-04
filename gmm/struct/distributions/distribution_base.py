# -*- coding: utf-8 -*-


class DistributionBase:

    def initialize(self, dtype):
        raise NotImplementedError()

    def get_parameters(self):
        raise NotImplementedError()

    def get_log_probabilities(self, data):
        raise NotImplementedError()

    def get_parameter_updaters(self, data, gamma_weighted, gamma_sum):
        raise NotImplementedError()
