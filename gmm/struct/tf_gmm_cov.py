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


class DiagonalCovariance(CovarianceBase):

    def __init__(self, dims, vector=None, prior=None):
        self.dims = dims
        self.vector = vector
        self.prior = prior
        self.has_prior = None

        self.tf_variance_vector = None
        self.tf_alpha = None
        self.tf_beta = None

    def initialize(self, dtype=tf.float64):
        if self.tf_variance_vector is None:
            if self.vector is not None:
                self.tf_variance_vector = tf.Variable(self.vector, dtype=dtype)
            else:
                self.tf_variance_vector = tf.Variable(tf.cast(tf.fill([self.dims], 1.0), dtype))

        if self.has_prior is None:
            if self.prior is not None:
                self.has_prior = True
                self.tf_alpha = tf.constant(self.prior["alpha"], dtype=dtype)
                self.tf_beta = tf.constant(self.prior["beta"], dtype=dtype)
            else:
                self.has_prior = False

    def get_matrix(self):
        return tf.diag(self.tf_variance_vector)

    def get_inv_quadratic_form(self, data, mean):
        tf_sq_distances = tf.squared_difference(data, tf.expand_dims(mean, 0))

        return tf.reduce_sum(tf_sq_distances / self.tf_variance_vector, 1)

    def get_log_determinant(self):
        return tf.reduce_sum(tf.log(self.tf_variance_vector))

    def get_prior_adjustment(self, original, gamma_sum):
        tf_adjusted = original
        tf_adjusted *= gamma_sum
        tf_adjusted += (2.0 * self.tf_beta)
        tf_adjusted /= gamma_sum + (2.0 * (self.tf_alpha + 1.0))

        return tf_adjusted

    def get_value_updater(self, data, new_mean, gamma_weighted, gamma_sum):
        tf_new_sq_distances = tf.squared_difference(data, tf.expand_dims(new_mean, 0))
        tf_new_variance = tf.reduce_sum(tf_new_sq_distances * tf.expand_dims(gamma_weighted, 1), 0)

        if self.has_prior:
            tf_new_variance = self.get_prior_adjustment(tf_new_variance, gamma_sum)

        return self.tf_variance_vector.assign(tf_new_variance)


class SparseCovariance(CovarianceBase):

    def __init__(self, dims, rank, baseline, eigvals=None, eigvecs=None, prior=None):
        self.dims = dims
        self.rank = rank
        self.baseline = baseline
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.prior = prior
        self.has_prior = None

        self.tf_baseline = None
        self.tf_eigvals = None
        self.tf_eigvecs = None
        self.tf_alpha = None
        self.tf_beta = None
        self.tf_rest = None

    def initialize(self, dtype=tf.float64):
        if self.tf_baseline is None:
            self.tf_baseline = tf.Variable(self.baseline, dtype)

        if self.tf_eigvals is None:
            if self.eigvals is not None:
                self.tf_eigvals = tf.Variable(self.eigvals, dtype)
            else:
                self.tf_eigvals = tf.Variable(tf.zeros([self.rank], dtype))

        if self.tf_eigvecs is None:
            if self.eigvecs is not None:
                self.tf_eigvecs = tf.Variable(self.eigvecs, dtype)
            else:
                self.tf_eigvecs = tf.Variable(tf.zeros([self.rank, self.dims], dtype))

        if self.has_prior is None:
            if self.prior is not None:
                self.has_prior = True
                self.tf_alpha = tf.constant(self.prior["alpha"], dtype=dtype)
                self.tf_beta = tf.constant(self.prior["beta"], dtype=dtype)
            else:
                self.has_prior = False

        if self.tf_rest is None:
            self.tf_rest = tf.constant(self.dims - self.rank, dtype=dtype)

    def get_matrix(self):
        tf_base_times_eye = tf.diag(tf.fill([self.dims], self.tf_baseline))
        tf_eig_vec_val = tf.matmul(tf.transpose(self.tf_eigvecs), tf.diag(self.tf_eigvals))
        tf_eig_vec_val_vec = tf.matmul(tf_eig_vec_val, self.tf_eigvecs)

        return tf_base_times_eye + tf_eig_vec_val_vec

    def get_inv_quadratic_form(self, data, mean):
        tf_differences = tf.subtract(data, tf.expand_dims(mean, 0))
        tf_diff_times_eig = tf.matmul(tf_differences, tf.transpose(self.tf_eigvecs))
        tf_factor = 1.0 / (self.tf_baseline + self.tf_eigvals) - 1.0 / self.tf_baseline

        tf_base_part = tf.reduce_sum(tf.square(tf_differences) / self.tf_baseline, 1)
        tf_eig_part = tf.reduce_sum(tf.square(tf_diff_times_eig) * tf_factor, 1)

        return tf_base_part + tf_eig_part

    def get_log_determinant(self):
        tf_rank_part = tf.reduce_sum(tf.log(self.tf_baseline + self.tf_eigvals))
        tf_rest_part = tf.log(self.tf_baseline) * self.tf_rest

        return tf_rank_part + tf_rest_part

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

        tf_s, tf_u, _ = tf.svd(tf_new_covariance)

        tf_required_eigvals = tf_s[:self.rank]
        tf_required_eigvecs = tf_u[:, :self.rank]

        tf_new_baseline = (tf.trace(tf_new_covariance) - tf.reduce_sum(tf_required_eigvals)) / self.tf_rest
        tf_new_eigvals = tf_required_eigvals - tf_new_baseline
        tf_new_eigvecs = tf.transpose(tf_required_eigvecs)

        return tf.group(
            self.tf_baseline.assign(tf_new_baseline),
            self.tf_eigvals.assign(tf_new_eigvals),
            self.tf_eigvecs.assign(tf_new_eigvecs)
        )


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
