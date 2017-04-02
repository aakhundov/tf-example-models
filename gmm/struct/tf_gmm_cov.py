import tensorflow as tf


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


class IsotropicCovariance(CovarianceBase):

    def __init__(self, dims, scalar=None, alpha=None, beta=None):
        self.dims = dims
        self.scalar = scalar
        self.alpha = alpha
        self.beta = beta

        self._variance_scalar = None
        self._prior = None
        self._alpha = None
        self._beta = None
        self._dims = None

    def initialize(self, dtype=tf.float64):
        if self._variance_scalar is None:
            if self.scalar is not None:
                self._variance_scalar = tf.Variable(self.scalar, dtype=dtype)
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

    def get_inv_quadratic_form(self, data, mean):
        sq_distances = tf.squared_difference(data, tf.expand_dims(mean, 0))
        sum_sq_distances = tf.reduce_sum(sq_distances, 1)

        return sum_sq_distances / self._variance_scalar

    def get_log_determinant(self):
        return self._dims * tf.log(self._variance_scalar)

    def get_prior_adjustment(self, original, gamma_sum):
        adjusted = original
        adjusted *= gamma_sum
        adjusted += (2.0 * self._beta)
        adjusted /= gamma_sum + (2.0 * (self._alpha + 1.0))

        return adjusted

    def get_value_updater(self, data, new_mean, gamma_weighted, gamma_sum):
        new_sq_distances = tf.squared_difference(data, tf.expand_dims(new_mean, 0))
        new_variance = tf.reduce_sum(new_sq_distances * tf.expand_dims(gamma_weighted, 1)) / self._dims

        if self._prior:
            new_variance = self.get_prior_adjustment(new_variance, gamma_sum)

        return self._variance_scalar.assign(new_variance)


class DiagonalCovariance(CovarianceBase):

    def __init__(self, dims, vector=None, alpha=None, beta=None):
        self.dims = dims
        self.vector = vector
        self.alpha = alpha
        self.beta = beta

        self._variance_vector = None
        self._prior = None
        self._alpha = None
        self._beta = None

    def initialize(self, dtype=tf.float64):
        if self._variance_vector is None:
            if self.vector is not None:
                self._variance_vector = tf.Variable(self.vector, dtype=dtype)
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

    def get_inv_quadratic_form(self, data, mean):
        sq_distances = tf.squared_difference(data, tf.expand_dims(mean, 0))

        return tf.reduce_sum(sq_distances / self._variance_vector, 1)

    def get_log_determinant(self):
        return tf.reduce_sum(tf.log(self._variance_vector))

    def get_prior_adjustment(self, original, gamma_sum):
        adjusted = original
        adjusted *= gamma_sum
        adjusted += (2.0 * self._beta)
        adjusted /= gamma_sum + (2.0 * (self._alpha + 1.0))

        return adjusted

    def get_value_updater(self, data, new_mean, gamma_weighted, gamma_sum):
        new_sq_distances = tf.squared_difference(data, tf.expand_dims(new_mean, 0))
        new_variance = tf.reduce_sum(new_sq_distances * tf.expand_dims(gamma_weighted, 1), 0)

        if self._prior:
            new_variance = self.get_prior_adjustment(new_variance, gamma_sum)

        return self._variance_vector.assign(new_variance)


class SparseCovariance(CovarianceBase):

    def __init__(self, dims, rank, baseline, eigvals=None, eigvecs=None, alpha=None, beta=None):
        self.dims = dims
        self.rank = rank
        self.baseline = baseline
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.alpha = alpha
        self.beta = beta

        self._baseline = None
        self._eigvals = None
        self._eigvecs = None
        self._prior = None
        self._alpha = None
        self._beta = None
        self._rest = None

    def initialize(self, dtype=tf.float64):
        if self._baseline is None:
            self._baseline = tf.Variable(self.baseline, dtype)

        if self._eigvals is None:
            if self.eigvals is not None:
                self._eigvals = tf.Variable(self.eigvals, dtype)
            else:
                self._eigvals = tf.Variable(tf.zeros([self.rank], dtype))

        if self._eigvecs is None:
            if self.eigvecs is not None:
                self._eigvecs = tf.Variable(self.eigvecs, dtype)
            else:
                self._eigvecs = tf.Variable(tf.zeros([self.rank, self.dims], dtype))

        if self._prior is None:
            if self.alpha is not None and self.beta is not None:
                self._prior = True
                self._alpha = tf.constant(self.alpha, dtype=dtype)
                self._beta = tf.constant(self.beta, dtype=dtype)
            else:
                self._prior = False

        if self._rest is None:
            self._rest = tf.constant(self.dims - self.rank, dtype=dtype)

    def get_matrix(self):
        base_times_eye = tf.diag(tf.fill([self.dims], self._baseline))
        eig_vec_val = tf.matmul(tf.transpose(self._eigvecs), tf.diag(self._eigvals))
        eig_vec_val_vec = tf.matmul(eig_vec_val, self._eigvecs)

        return base_times_eye + eig_vec_val_vec

    def get_inv_quadratic_form(self, data, mean):
        differences = tf.subtract(data, tf.expand_dims(mean, 0))
        diff_times_eig = tf.matmul(differences, tf.transpose(self._eigvecs))
        factor = 1.0 / (self._baseline + self._eigvals) - 1.0 / self._baseline

        base_part = tf.reduce_sum(tf.square(differences) / self._baseline, 1)
        eig_part = tf.reduce_sum(tf.square(diff_times_eig) * factor, 1)

        return base_part + eig_part

    def get_log_determinant(self):
        rank_part = tf.reduce_sum(tf.log(self._baseline + self._eigvals))
        rest_part = tf.log(self._baseline) * self._rest

        return rank_part + rest_part

    def get_prior_adjustment(self, original, gamma_sum):
        adjusted = original
        adjusted *= gamma_sum
        adjusted += tf.diag(tf.fill([self.dims], 2.0 * self._beta))
        adjusted /= gamma_sum + (2.0 * (self._alpha + 1.0))

        return adjusted

    def get_value_updater(self, data, new_mean, gamma_weighted, gamma_sum):
        new_differences = tf.subtract(data, tf.expand_dims(new_mean, 0))
        sq_dist_matrix = tf.matmul(tf.expand_dims(new_differences, 2), tf.expand_dims(new_differences, 1))
        new_covariance = tf.reduce_sum(sq_dist_matrix * tf.expand_dims(tf.expand_dims(gamma_weighted, 1), 2), 0)

        if self._prior:
            new_covariance = self.get_prior_adjustment(new_covariance, gamma_sum)

        s, u, _ = tf.svd(new_covariance)

        required_eigvals = s[:self.rank]
        required_eigvecs = u[:, :self.rank]

        new_baseline = (tf.trace(new_covariance) - tf.reduce_sum(required_eigvals)) / self._rest
        new_eigvals = required_eigvals - new_baseline
        new_eigvecs = tf.transpose(required_eigvecs)

        return tf.group(
            self._baseline.assign(new_baseline),
            self._eigvals.assign(new_eigvals),
            self._eigvecs.assign(new_eigvecs)
        )


class FullCovariance(CovarianceBase):

    def __init__(self, dims, matrix=None, alpha=None, beta=None, approx_log_det=False):
        self.dims = dims
        self.matrix = matrix
        self.alpha = alpha
        self.beta = beta

        self.approx_log_det = approx_log_det

        self._covariance_matrix = None
        self._prior = None
        self._alpha = None
        self._beta = None

    def initialize(self, dtype=tf.float64):
        if self._covariance_matrix is None:
            if self.matrix is not None:
                self._covariance_matrix = tf.Variable(self.matrix, dtype=dtype)
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

    def get_inv_quadratic_form(self, data, mean):
        differences = tf.subtract(data, tf.expand_dims(mean, 0))
        diff_times_inv_cov = tf.matmul(differences, tf.matrix_inverse(self._covariance_matrix))

        return tf.reduce_sum(diff_times_inv_cov * differences, 1)

    def get_log_determinant(self):
        if self.approx_log_det:
            return tf.reduce_sum(tf.log(tf.diag_part(self._covariance_matrix)))
        else:
            return tf.log(tf.matrix_determinant(self._covariance_matrix))

    def get_prior_adjustment(self, original, gamma_sum):
        adjusted = original
        adjusted *= gamma_sum
        adjusted += tf.diag(tf.fill([self.dims], 2.0 * self._beta))
        adjusted /= gamma_sum + (2.0 * (self._alpha + 1.0))

        return adjusted

    def get_value_updater(self, data, new_mean, gamma_weighted, gamma_sum):
        new_differences = tf.subtract(data, tf.expand_dims(new_mean, 0))
        sq_dist_matrix = tf.matmul(tf.expand_dims(new_differences, 2), tf.expand_dims(new_differences, 1))
        new_covariance = tf.reduce_sum(sq_dist_matrix * tf.expand_dims(tf.expand_dims(gamma_weighted, 1), 2), 0)

        if self._prior:
            new_covariance = self.get_prior_adjustment(new_covariance, gamma_sum)

        return self._covariance_matrix.assign(new_covariance)
