import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
from torch import from_numpy
from keras.datasets import mnist, cifar10

def get_data(dataset_type, path=None):
    if dataset_type == "MNIST":
        train, test = mnist.load_data()
    elif dataset_type == "CIFAR10":
        train, test = cifar10.load_data()

    train_x, train_y = from_numpy(train[0]), from_numpy(train[1])
    test_x, test_y = from_numpy(test[0]), from_numpy(test[1])
    train, test = (train_x, train_y), (test_x, test_y)

    return train, test

def normalized_euclidian_transform():
    (cifar_train_data, cifar_train_targets), (cifar_test_data, cifar_test_targets) = cifar10.load_data()

    norm_layer = Normalization()
    norm_layer.adapt(cifar_train_data)
    cifar_train_data = norm_layer(cifar_train_data)
    cifar_test_data = norm_layer(cifar_test_data)

    input_shape = cifar_train_data.shape[1:]
    inputs = Input(shape=input_shape)

    distance = Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x-0.5), axis=-1)))(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=distance)

    cifar_train_data = model.predict(cifar_train_data)
    cifar_train_targets = np.squeeze(cifar_train_targets, axis=-1)
    cifar_test_data = model.predict(cifar_test_data)
    cifar_test_targets = np.squeeze(cifar_test_targets, axis=-1)

    return (cifar_train_data, cifar_train_targets), (cifar_test_data, cifar_test_targets)

def log_class_counts(y_train, subset_ID_map, log=False):
    cls_counts = {}

    for subset_i, ID in subset_ID_map.items():
        unq, unq_cnt = np.unique(y_train[ID], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        cls_counts[subset_i] = tmp

    if log:
        logging.debug('Label distributions: %s' % str(cls_counts))

    return cls_counts

def map_to_prob(y_train, subset_map):
    counts = log_class_counts(y_train, subset_map)

    values = [np.array([counts.get(k).get(key) for key in counts.get(k)]) for k in counts]
    probs = [values[j] / values[j].sum() for j in range(len(values))]

    return probs

def tensor_to_csv(x_train, y_train, subset_map, epoch_num):
    for key in subset_map:
        x, y = x_train[subset_map.get(key)], y_train[subset_map.get(key)]
        df_x = pd.DataFrame(x.tolist())
        df_y = pd.DataFrame(y.tolist())
        df_x['targets'] = df_y
        df_x.rename(columns={0: 'data', "targets": "targets"})
        df_x.to_csv(f'subsets/epoch_{epoch_num}_subset_{key + 1}.csv', index=False)

def getDivs(N):
    factors = {1}
    maxP  = int(N**0.5)
    p,inc = 2,1
    while p <= maxP:
        while N%p==0:
            factors.update([f*p for f in factors])
            N //= p
            maxP = int(N**0.5)
        p,inc = p+inc,2
    if N>1:
        factors.update([f*N for f in factors])
    return sorted(factors)

def vis_datasets(path, epoch, y_train, subset_ID_map, mode_plot, mode_partitioning, alpha, save=False):
    counts = log_class_counts(y_train, subset_ID_map)

    values = [np.array([counts.get(k).get(key) for key in counts.get(k)]) for k in counts]
    values_normalized = [values[j] / values[j].sum() for j in range(len(values))]

    n_clients = len(subset_ID_map)

    title_formats = {
        "hetero-dir": "A distribution-based heterogeneous partitioning X~Dir({alpha}) with {n_clients} subsets",
        "hetero-gaussian": "A distribution-based Gaussian heterogeneous partitioning σ={alpha} with {n_clients} subsets",
        "homo": "A homogeneous partitioning with {n_clients} subsets",
    }
    main_title = title_formats.get(mode_partitioning, "")
    main_title = main_title.format(alpha=alpha, n_clients=n_clients)

    subtitle_formats = {
        "hetero_dir": "α_{epoch}={alpha}",
        "hetero-gaussian": "σ_{epoch}={alpha}",
        "homo" : ""
    }

    subtitle = subtitle_formats.get(mode_partitioning, "")
    subtitle = subtitle.format(epoch=epoch, alpha=alpha)

    if mode_plot == "heatmap":
        ax = sns.heatmap(pd.DataFrame(values_normalized), vmin=0, vmax=1, cmap=sns.cm.rocket_r)
        ax.set(xlabel="Labels", ylabel="Clients", title=main_title)
        plt.show()
    elif mode_plot == "histogram":
        dim_x = 2
        dim_y = 5
        K = len(set(y_train.tolist()))
        fig, axes = plt.subplots(dim_y, dim_x)
        # TODO: orient hard-coded values around 'K'
        for j in range(len(values)):
            plt.figure(figsize=(5, 3), dpi=300)
            plt.hist(values[j], [(i - 0.5) / 2 for i in range( 2 *K)], label="Sampled dist")
            x = np.arange(-0.5, 9.5, 0.1)
            plt.xticks([i for i in range(K)])
            plt.xlabel("Label")
            plt.xlim([-1, 10])
            plt.ylabel("Entries")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()


def vis_divergence(evol_stat, mode_plot, main_title, xlab):
    def linplot(stats):
        plt.figure(figsize=(5, 3), dpi=300)
        plt.plot(np.arange(len(stats)), stats)
        plt.xlabel(xlab)
        plt.suptitle(main_title, fontsize='small')
        plt.show()

    mode_dict = {
        "lp": linplot
    }

    try:
        plot_func = mode_dict[mode_plot]
    except KeyError:
        raise ValueError(f"Invalid mode: {mode_plot}")

    plot_func(evol_stat)

def plot_curve(epochs, hist, list_of_metrics):
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")

  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)

  plt.legend()