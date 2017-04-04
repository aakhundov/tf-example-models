# -*- coding: utf-8 -*-


class CovarianceBase:

    def initialize(self, dtype):
        raise NotImplementedError()

    def get_matrix(self):
        raise NotImplementedError()

    def get_inv_quadratic_form(self, data, mean):
        raise NotImplementedError()

    def get_log_determinant(self):
        raise NotImplementedError()

    def get_value_updater(self, data, new_mean, gamma_weighted, gamma_sum):
        raise NotImplementedError()
