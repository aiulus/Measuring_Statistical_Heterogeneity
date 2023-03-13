import numpy as np
import math
import random
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import kstest, anderson_ksamp, cumfreq, ks_2samp, cramervonmises, chisquare, entropy, \
    wasserstein_distance
from src.utils.utils import get_data, getDivs, map_to_prob, vis_divergence, tensor_to_csv
from statsmodels.distributions.empirical_distribution import ECDF
from src.utils.partitioning import partition


def distance(y_train, subset_map, mode):
    def kl_divergence(p, q):
        return sum(p[i] * math.log2(p[i] / q[i]) for i in range(len(p)) if q[i] != 0 and p[i] != 0)

    # Jensen-Shannon Divergence
    def js_divergence(p, q):
        m = 0.5 * (p + q)
        return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

    # Gini Coefficient
    def gini(x):
        total = 0
        for i, xi in enumerate(x[:-1], 1):
            total += np.sum(np.abs(xi - x[i:]))
        return total / (len(x) ** 2 * np.mean(x))

    N = y_train.shape[0]
    probs_original, probs = map_to_prob(y_train, {0: np.arange(N)})[0], map_to_prob(y_train, subset_map)
    stats, pvals = [], []

    mode_dict = {
        "kolmogorov-smirnov": lambda x: kstest(x, probs_original),
        "empirical": lambda x: ks_2samp(ECDF(x)(x), ECDF(probs_original)(probs_original)),
        "cramer-von-mises": lambda x: cramervonmises(ECDF(x), ECDF(probs_original)),
        "pearson-chi-squared": lambda x: chisquare(x * N, probs_original * N),
        "anderson-darling": anderson_ksamp
    }

    mode_entropy = {
        "kl": lambda x: kl_divergence(x, probs_original),
        "js": lambda x: js_divergence(x, probs_original),
        "wd": lambda x: wasserstein_distance(x, probs_original),
        "gini": gini
    }

    if (mode in mode_entropy):
        try:
            test_func = mode_entropy[mode]
        except KeyError:
            raise ValueError(f"Invalid mode: {mode}")

        stats, pvals = [test_func(probs[j]) for j in range(len(probs))], []
    else:
        try:
            test_func = mode_dict[mode]
        except KeyError:
            raise ValueError(f"Invalid mode: {mode}")

        stats, pvals = zip(*[test_func(probs[j]) for j in range(len(probs))])

    return stats, pvals


# TODO: entry point 4
def run_experiment(dataset_type, alpha_vector, path, mode_part, mode_test, mode_plot, n_clients, plot=False,
                   logpath=None, save=False):
    train, test = get_data(dataset_type, path)
    x_train, y_train, x_test, y_test = train[0], train[1], test[0], test[1]
    evol_stat, evol_pval = [], []

    for j in range(len(alpha_vector)):
        a_j = alpha_vector[j]
        subset_map = partition(train, mode_part, n_clients, a_j)
        if save:
            tensor_to_csv(x_train, y_train, subset_map, j)
        stats, pvals = distance(y_train, subset_map, mode_test)
        mean_teststat_j = np.mean(stats)
        mean_pval_j = np.mean(pvals)
        evol_stat.append(mean_teststat_j)
        evol_pval.append(mean_pval_j)

    title_formats_part = {
        "hetero-dir": "Mean divergence from original distribution under heterogeneous partitioning via Dirichlet distribution",
        "hetero-gaussian": "Mean divergence from original distribution under heterogeneous partitioning via Gaussian distribution",
        "homo": "A homogeneous partitioning with {n_clients} subsets",
        "quant": "A quantity-based heterogeneous partitioning with {n_clients} subsets"
    }

    title_formats_test = {
        "kolmogorov-smirnov": "Test Statistic: Kolmogorov-Smirnov",
        "empirical": "Test Statistic: Empirical Distribution",
        "kl": "Test Statistic: Kullback-Leibler Divergence (Entropy-Based)",
        "js": "Test Statistic: Jensen-Shannon Divergence (Entropy-Based)",
        "wd": "Test Statistic: Wasserstein Distance",
        "gini": "Test Statistic: Gini Coefficient"
    }

    main_title = title_formats_part.get(mode_part, "")
    main_title = main_title.format(n_clients=n_clients)
    appendage = title_formats_test.get(mode_test, "")
    main_title = dataset_type + ': ' + main_title + '\n' + appendage

    if mode_part == "quant":
        l, u = tuple(round(x, 1) for x in (min(alpha_vector), max(alpha_vector)))
        xlab = f'[{u}:{l}], set of divisors for {u}'
    elif len(alpha_vector) <= 15:
        xlab = [f'α_{j}={alpha}' for j, alpha in enumerate(alpha_vector)]
    else:
        l, u, steps = tuple(round(x, 1) for x in (
            min(alpha_vector), max(alpha_vector), (max(alpha_vector) - min(alpha_vector)) / (len(alpha_vector) - 1)))
        xlab = f'[{l}:{u}], step size = {steps}'

    if plot:
        vis_divergence(evol_stat, mode_plot, main_title, xlab)

    return (evol_stat, evol_pval)


# TODO: entry point 3
def multi_plot(dataset_type, alpha_vector, path, mode_part, mode_test, mode_plot, n_clients, runs):
    vals = []
    for j in range(runs):
        stats = np.array(
            run_experiment(dataset_type, alpha_vector, path, mode_part, mode_test, mode_plot, n_clients, plot=False)[0])
        vals.append(stats)

    p = pd.DataFrame(vals)
    means = []
    stds = []
    for j in range(p.shape[1]):
        col = np.array(p[j])
        means.append(col.mean())
        # sigma_j = math.sqrt(sum([math.pow(x - np.array(p[j]).mean(), 2) for x in p[j]])/len(p[j]))
        stds.append(np.array(p[j]).std())
    means, stds = np.array(means), np.array(stds)

    part_dict = {
        "hetero-dir": "Dirichlet-Partitioning",
        "hetero-gaussian": "Gaussian Partitioning",
        "homo": "Homogeneous Partitioning",
        "quant": "Quantity-based Heterogeneous Partitioning"
    }

    test_dict = {
        "kolmogorov-smirnov": "Test Statistic: Kolmogorov-Smirnov",
        "empirical": "Test Statistic: Empirical Distribution",
        "kl": "Test Statistic: Kullback-Leibler Divergence (Entropy-Based)",
        "js": "Test Statistic: Jensen-Shannon Divergence (Entropy-Based)",
        "wd": "Test Statistic: Wasserstein Distance",
        "gini": "Test Statistic: Gini Coefficient"
    }

    main_title = f'Evolution of divergence over {runs} epochs.\n ' + part_dict.get(mode_part,
                                                                                   "") + ", " + test_dict.get(mode_test,
                                                                                                              "")

    plt.figure(figsize=(5, 3), dpi=300)
    plt.plot(np.arange(len(means)), means)
    plt.fill_between(np.arange(len(means)), means - stds, means + stds, color='gray', alpha=0.2)
    plt.grid(linestyle='-', linewidth=0.5)
    plt.suptitle(main_title, fontsize='small')
    plt.show()
