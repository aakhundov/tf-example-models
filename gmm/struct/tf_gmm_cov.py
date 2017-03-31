import tensorflow as tf


class CovarianceBase:
    def initialize(self, dtype):
        raise NotImplementedError()

    def get_matrix(self):
        raise NotImplementedError()

    def get_quadratic_form(self, data, mean):
        raise NotImplementedError()

    def get_log_determinant(self):
        raise NotImplementedError()

    def get_prior_adjustment(self, variance, gamma_sum):
        raise NotImplementedError()

    def get_value_updater(self, data, new_mean, gamma_weighted, gamma_sum):
        raise NotImplementedError()


class IsotropicCovariance(CovarianceBase):
    def __init__(self, dims, initial=None, alpha=None, beta=None):
        self.dims = dims
        self.initial = initial
        self.alpha = alpha
        self.beta = beta

        self._variance_scalar = None
        self._prior = None
        self._alpha = None
        self._beta = None
        self._dims = None

    def initialize(self, dtype=tf.float64):
        if self._variance_scalar is None:
            if self.initial is not None:
                self._variance_scalar = tf.Variable(self.initial, dtype=dtype)
            else:
                self._variance_scalar = tf.Variable(1.0, dtype=dtype)

        if self._prior is None:
            if self.alpha is not None and self.beta is not None:
                self._prior = True
                self._alpha = tf.constant(self.alpha, dtype=dtype)
                self._beta = tf.constant(self.beta, dtype=dtype)
            else:
                self._prior = False

        self._dims = tf.constant(self.dims, dtype=dtype)

    def get_matrix(self):
        return tf.diag(tf.fill([self.dims], self._variance_scalar))

    def get_quadratic_form(self, data, mean):
        sq_distances = tf.squared_difference(data, tf.expand_dims(mean, 0))
        sum_sq_distances = tf.reduce_sum(sq_distances, 1)

        return sum_sq_distances / self._variance_scalar

    def get_log_determinant(self):
        return self._dims * tf.log(self._variance_scalar)

    def get_prior_adjustment(self, variance, gamma_sum):
        adjusted_variance = variance
        adjusted_variance *= gamma_sum
        adjusted_variance += (2.0 * self._beta)
        adjusted_variance /= gamma_sum + (2.0 * (self._alpha + 1.0))

        return adjusted_variance

    def get_value_updater(self, data, new_mean, gamma_weighted, gamma_sum):
        new_sq_distances = tf.squared_difference(data, tf.expand_dims(new_mean, 0))
        new_variance = tf.reduce_sum(new_sq_distances * tf.expand_dims(gamma_weighted, 1)) / self._dims

        if self._prior:
            new_variance = self.get_prior_adjustment(new_variance, gamma_sum)

        return self._variance_scalar.assign(new_variance)


class DiagonalCovariance(CovarianceBase):
    def __init__(self, dims, initial=None, alpha=None, beta=None):
        self.dims = dims
        self.initial = initial
        self.alpha = alpha
        self.beta = beta

        self._variance_vector = None
        self._prior = None
        self._alpha = None
        self._beta = None

    def initialize(self, dtype=tf.float64):
        if self._variance_vector is None:
            if self.initial is not None:
                self._variance_vector = tf.Variable(self.initial, dtype=dtype)
            else:
                self._variance_vector = tf.Variable(tf.cast(tf.fill([self.dims], 1.0), dtype))

        if self._prior is None:
            if self.alpha is not None and self.beta is not None:
                self._prior = True
                self._alpha = tf.constant(self.alpha, dtype=dtype)
                self._beta = tf.constant(self.beta, dtype=dtype)
            else:
                self._prior = False

    def get_matrix(self):
        return tf.diag(self._variance_vector)

    def get_quadratic_form(self, data, mean):
        sq_distances = tf.squared_difference(data, tf.expand_dims(mean, 0))

        return tf.reduce_sum(sq_distances / self._variance_vector, 1)

    def get_log_determinant(self):
        return tf.reduce_sum(tf.log(self._variance_vector))

    def get_prior_adjustment(self, variance, gamma_sum):
        adjusted_variance = variance
        adjusted_variance *= gamma_sum
        adjusted_variance += (2.0 * self._beta)
        adjusted_variance /= gamma_sum + (2.0 * (self._alpha + 1.0))

        return adjusted_variance

    def get_value_updater(self, data, new_mean, gamma_weighted, gamma_sum):
        new_sq_distances = tf.squared_difference(data, tf.expand_dims(new_mean, 0))
        new_variance = tf.reduce_sum(new_sq_distances * tf.expand_dims(gamma_weighted, 1), 0)

        if self._prior:
            new_variance = self.get_prior_adjustment(new_variance, gamma_sum)

        return self._variance_vector.assign(new_variance)


class FullCovariance(CovarianceBase):
    def __init__(self, dims, initial=None, alpha=None, beta=None):
        self.dims = dims
        self.initial = initial
        self.alpha = alpha
        self.beta = beta

        self._covariance_matrix = None
        self._prior = None
        self._alpha = None
        self._beta = None

    def initialize(self, dtype=tf.float64):
        if self._covariance_matrix is None:
            if self.initial is not None:
                self._covariance_matrix = tf.Variable(self.initial, dtype=dtype)
            else:
                self._covariance_matrix = tf.Variable(tf.diag(tf.cast(tf.fill([self.dims], 1.0), dtype)))

        if self._prior is None:
            if self.alpha is not None and self.beta is not None:
                self._prior = True
                self._alpha = tf.constant(self.alpha, dtype=dtype)
                self._beta = tf.constant(self.beta, dtype=dtype)
            else:
                self._prior = False

    def get_matrix(self):
        return self._covariance_matrix

    def get_quadratic_form(self, data, mean):
        differences = tf.subtract(data, tf.expand_dims(mean, 0))
        diff_times_inv_cov = tf.matmul(differences, tf.matrix_inverse(self._covariance_matrix))

        return tf.reduce_sum(diff_times_inv_cov * differences, 1)

    def get_log_determinant(self):
        return tf.log(tf.matrix_determinant(self._covariance_matrix))

    def get_prior_adjustment(self, covariance, gamma_sum):
        adjusted_covariance = covariance
        adjusted_covariance *= gamma_sum
        adjusted_covariance += tf.diag(tf.fill([self.dims], 2.0 * self._beta))
        adjusted_covariance /= gamma_sum + (2.0 * (self._alpha + 1.0))

        return adjusted_covariance

    def get_value_updater(self, data, new_mean, gamma_weighted, gamma_sum):
        new_differences = tf.subtract(data, tf.expand_dims(new_mean, 0))
        sq_dist_matrix = tf.matmul(tf.expand_dims(new_differences, 2), tf.expand_dims(new_differences, 1))
        new_covariance = tf.reduce_sum(sq_dist_matrix * tf.expand_dims(tf.expand_dims(gamma_weighted, 1), 2), 0)

        if self._prior:
            new_covariance = self.get_prior_adjustment(new_covariance, gamma_sum)

        return self._covariance_matrix.assign(new_covariance)
