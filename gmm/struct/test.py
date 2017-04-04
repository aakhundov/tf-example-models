import numpy as np
import tensorflow as tf

import covariances
import distributions
import models

import utils


def feedback_sub(step, current_log_likelihood, difference):
    if step > 0:
        print("{0}:\tmean-log-likelihood {1:.8f}\tdifference {2}".format(
            step, current_log_likelihood, difference))
    else:
        print("{0}:\tmean-log-likelihood {1:.8f}".format(
            step, current_log_likelihood))


def test_gmm(num_points, components, dimensions, rank, tolerance, training_steps, cluster=None):
    print("Generating data...")
    synthetic_data, true_means, true_covariances, true_weights, responsibilities = utils.generate_gmm_data(
        num_points, components, dimensions, seed=10, diagonal=False)

    print("Computing avg. covariance...")
    avg_data_variance = np.var(synthetic_data, axis=0).sum() / components / dimensions

    print("Initializing components...")
    mixture_components = []
    for c in range(components):
        mixture_components.append(
            distributions.GaussianDistribution(
                dims=dimensions,
                mean=synthetic_data[c],
                # covariance=covariances.IsotropicCovariance(
                #     dimensions,
                #     scalar=avg_data_variance,
                #     prior={"alpha": 1.0, "beta": 1.0}
                # ),
                # covariance=covariances.DiagonalCovariance(
                #     dimensions,
                #     vector=np.full((dimensions,), avg_data_variance),
                #     prior={"alpha": 1.0, "beta": 1.0}
                # ),
                covariance=covariances.SparseCovariance(
                    dimensions, rank,
                    baseline=avg_data_variance,
                    prior={"alpha": 1.0, "beta": 1.0}
                ),
                # covariance=covariances.FullCovariance(
                #     dimensions,
                #     matrix=np.diag(np.full((dimensions,), avg_data_variance)),
                #     prior={"alpha": 1.0, "beta": 1.0}
                # ),
            )
        )

    print("Initializing model...")
    gmm = models.MixtureModel(synthetic_data, mixture_components, cluster=cluster)

    print("Training model...\n")
    result = gmm.train(tolerance=tolerance, max_steps=training_steps, feedback=feedback_sub)

    final_means = np.stack([result[2][i][0] for i in range(components)])
    final_covariances = np.stack([result[2][i][1] for i in range(components)])

    utils.plot_fitted_data(
        synthetic_data[:, :2],
        final_means[:, :2], final_covariances[:, :2, :2],
        true_means[:, :2], true_covariances[:, :2, :2]
    )

    return result


def test_cmm(num_points, components, dimensions, tolerance, training_steps, cluster=None):
    print("Generating data...")
    synthetic_data, val_counts, true_means, true_weights, responsibilities = utils.generate_cmm_data(
        num_points, components, dimensions, seed=10, count_range=(2, 10)
    )

    print("Initializing components...")
    mixture_components = []
    for c in range(components):
        mixture_components.append(
            distributions.CategoricalDistribution(
                val_counts
            )
        )

    print("Initializing model...")
    gmm = models.MixtureModel(synthetic_data, mixture_components, cluster=cluster)

    print("Training model...\n")
    result = gmm.train(tolerance=tolerance, max_steps=training_steps, feedback=feedback_sub)

    return result


def test_cgmm(num_points, components, g_dimensions, g_rank, c_dimensions, tolerance, training_steps, cluster=None):
    print("Generating data...")
    c_data, g_data, c_counts, c_means, g_means, g_covariances, \
        true_weights, responsibilities = utils.generate_cgmm_data(
            num_points, components, c_dimensions, g_dimensions, seed=20)

    print("Computing avg. covariance...")
    avg_g_data_variance = np.var(g_data, axis=0).sum() / components / g_dimensions

    print("Initializing components...")
    mixture_components = [
        distributions.ProductDistribution([
            distributions.GaussianDistribution(
                dims=g_dimensions,
                mean=g_data[comp],
                # covariance=covariances.IsotropicCovariance(
                #     g_dimensions,
                #     scalar=avg_g_data_variance,
                #     prior={"alpha": 1.0, "beta": 1.0}
                # ),
                # covariance=covariances.DiagonalCovariance(
                #     g_dimensions,
                #     vector=np.full((g_dimensions,), avg_g_data_variance),
                #     prior={"alpha": 1.0, "beta": 1.0}
                # ),
                covariance=covariances.SparseCovariance(
                    g_dimensions, g_rank,
                    baseline=avg_g_data_variance,
                    prior={"alpha": 1.0, "beta": 1.0}
                ),
                # covariance=covariances.FullCovariance(
                #     g_dimensions,
                #     matrix=np.diag(np.full((g_dimensions,), avg_g_data_variance)),
                #     prior={"alpha": 1.0, "beta": 1.0}
                # ),
            ),
            distributions.CategoricalDistribution(
                c_counts
            )
        ])
        for comp in range(components)
    ]

    print("Initializing model...")
    gmm = models.MixtureModel([g_data, c_data], mixture_components, cluster=cluster)

    print("Training model...\n")
    result = gmm.train(tolerance=tolerance, max_steps=training_steps, feedback=feedback_sub)

    final_g_means = np.stack([result[2][i][0][0] for i in range(components)])
    final_g_covariances = np.stack([result[2][i][0][1] for i in range(components)])

    utils.plot_fitted_data(
        g_data[:, :2],
        final_g_means[:, :2], final_g_covariances[:, :2, :2],
        g_means[:, :2], g_covariances[:, :2, :2]
    )

    return result


G_RANK = 1
G_DIMENSIONS = 2
C_DIMENSIONS = 2
COMPONENTS = 10
NUM_POINTS = 10000

TRAINING_STEPS = 1000
TOLERANCE = 10e-6

cluster_spec = tf.train.ClusterSpec({
    "master": ["localhost:2222"],
    "worker": ["localhost:2223", "localhost:2224"]
})

# test_gmm(NUM_POINTS, COMPONENTS, G_DIMENSIONS, G_RANK, TOLERANCE, TRAINING_STEPS)
# test_cmm(NUM_POINTS, COMPONENTS, C_DIMENSIONS, TOLERANCE, TRAINING_STEPS)
test_cgmm(NUM_POINTS, COMPONENTS, G_DIMENSIONS, G_RANK, C_DIMENSIONS, TOLERANCE, TRAINING_STEPS)
