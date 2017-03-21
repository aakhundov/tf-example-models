import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat


def _generate_covariances(components, dimensions, diagonal=False):
    """Generates a batch of random positive definite covariance matrices"""
    covariances = np.zeros((components, dimensions, dimensions))

    if diagonal:
        for i in range(components):
            covariances[i] = np.diag(np.abs(np.random.normal(size=[dimensions])))
    else:
        for i in range(components):
            while np.any(np.linalg.eigvals(covariances[i]) <= 0):
                covariances[i] = np.random.normal(size=[dimensions, dimensions])
                np.fill_diagonal(covariances[i], np.abs(np.diag(covariances[i])))
                covariances[i] = (covariances[i] + covariances[i].T) / 2.0

    return covariances


def generate_gmm_data(size, components, dimensions, seed, diagonal=False):
    """Generates synthetic data of a given size from a random GMM"""
    np.random.seed(seed)

    means = np.random.normal(size=[components, dimensions]) * 10
    covariances = _generate_covariances(components, dimensions, diagonal)
    weights = np.abs(np.random.normal(size=[components]))
    weights /= np.sum(weights)

    result = np.empty((size, dimensions), dtype=np.float32)
    responsibilities = np.empty((size,), dtype=np.int32)
    component_array = np.array(range(components))

    for i in range(size):
        comp = np.random.choice(component_array, p=weights)

        responsibilities[i] = comp
        result[i] = np.random.multivariate_normal(
            means[comp], covariances[comp]
        )

    np.random.seed()

    return result, means, covariances, weights, responsibilities


def _plot_gaussian(mean, covariance, color, zorder=0):
    """Plots the mean and 2-std ellipse of a given Gaussian"""
    plt.plot(mean[0], mean[1], color[0] + ".", zorder=zorder)

    if covariance.ndim == 1:
        covariance = np.diag(covariance)

    radius = np.sqrt(5.991)
    eigvals, eigvecs = np.linalg.eig(covariance)
    axis = np.sqrt(eigvals) * radius
    slope = eigvecs[1][0] / eigvecs[1][1]
    angle = 180.0 * np.arctan(slope) / np.pi

    plt.axes().add_artist(pat.Ellipse(
        mean, 2 * axis[0], 2 * axis[1], angle=angle,
        fill=False, color=color, linewidth=1, zorder=zorder
    ))


def plot_fitted_data(data, means, covariances, true_means=None, true_variances=None):
    """Plots the data and given Gaussian components"""
    plt.plot(data[:, 0], data[:, 1], "b.", markersize=0.5, zorder=0)

    if true_means is not None:
        for i in range(len(true_means)):
            _plot_gaussian(true_means[i], true_variances[i], "green", 1)

    for i in range(len(means)):
        _plot_gaussian(means[i], covariances[i], "red", 2)

    plt.show()
