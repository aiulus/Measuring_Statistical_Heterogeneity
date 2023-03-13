import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras import optimizers
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical, plot_model
from keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
# from ann_visualizer.visualize import ann_viz

from src.utils.utils import getDivs, tensor_to_csv, plot_curve, normalized_euclidian_transform
from src.utils.partitioning import partition


def get_architecture(dataset_type, model_type):
    cnn_layers = {
        "MNIST": [Conv2D(filters=64,
                         kernel_size=3,
                         activation='relu',
                         input_shape=(28, 28, 1)),
                  MaxPooling2D(2),
                  Conv2D(filters=64,
                         kernel_size=3,
                         activation='relu'),
                  MaxPooling2D(2),
                  Conv2D(filters=64,
                         kernel_size=3,
                         activation='relu'),
                  Flatten(),
                  Dropout(0.2),
                  Dense(10, activation='softmax')
                  ],
        "CIFAR10": [Conv2D(32, (3, 3),
                           activation='relu',
                           input_shape=(32, 32, 3)),
                    MaxPooling2D((2, 2)),
                    Conv2D(64, (3, 3),
                           activation='relu'),
                    MaxPooling2D((2, 2)),
                    Conv2D(64, (3, 3),
                           activation='relu'),
                    Flatten(),
                    Dense(64, activation='relu'),
                    Dense(10)]
    }

    cnn_loss = {
        "MNIST": CategoricalCrossentropy(),
        "CIFAR10": CategoricalCrossentropy()
    }

    # TODO: Find best configuration for CIFAR10
    nn_layers = {
        "MNIST": [
            Flatten(input_shape=(28, 28)),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dropout(rate=0.4),
            Dense(units=10, activation='softmax')
        ],
        "CIFAR10": [
            Flatten(input_shape=(32, 32)),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dropout(rate=0.4),
            Dense(units=10, activation='softmax')
        ]
    }

    nn_loss = {
        "MNIST": CategoricalCrossentropy(),
        "CIFAR10": CategoricalCrossentropy()
    }

    if model_type == "nn":
        return nn_layers.get(dataset_type), nn_loss.get(dataset_type)
    elif model_type == "cnn":
        return cnn_layers.get(dataset_type), cnn_loss.get(dataset_type)
    else:
        raise ValueError(f"Invalid mode: {model_type}")


def build_model(dataset_type, model_type, data, n_epochs, lrate, batch_size=None):
    train_data, train_targets, test_data, test_targets = data

    layers, loss_function = get_architecture(dataset_type, model_type)

    model = Sequential()

    for layer in layers:
        model.add(layer)

    model.compile(loss=loss_function,
                  optimizer=optimizers.Adam(learning_rate=lrate),
                  metrics=['accuracy'])

    # TODO: batch_size --> SGD
    # 60000/32 = 1875 datapoints per epoch
    history = model.fit(train_data, train_targets,
                        epochs=n_epochs,
                        validation_data=(test_data, test_targets), batch_size=batch_size)
    # batch_size=4000
    loss, accuracy = model.evaluate(test_data, test_targets, batch_size=batch_size)

    return accuracy, loss, history.epoch, pd.DataFrame(history.history)


# TODO: entry point 2
def run_experiment(model_type, dataset_type, partitioning_type, n_clients, alpha, n_epochs, lrate=0.003,
                   batch_size=None):
    dataset_loaders = {
        "MNIST": mnist.load_data,
        "CIFAR10": normalized_euclidian_transform if model_type == "nn" else cifar10.load_data
    }

    try:
        loader = dataset_loaders[dataset_type]
    except KeyError:
        raise ValueError(f"Invalid mode: {dataset_type}")

    train, test = loader()
    test_data, test_targets = test[0], test[1]

    test_targets = to_categorical(test_targets, num_classes=10)
    if dataset_type == "MNIST":
        test_data = test_data.reshape((-1,) + test_data.shape[1:3] + (1,))

    subset_map = partition(dataset_type, partitioning_type, n_clients, alpha)
    accuracies = []

    for j in range(len(subset_map)):
        subset_j_data = train[0][subset_map[j]]
        subset_j_targets = train[1][subset_map[j]]

        # convert to one-hot vector
        # subset_j_targets = to_categorical(subset_j_targets)
        subset_j_targets = to_categorical(subset_j_targets, num_classes=10)

        if dataset_type == "MNIST":
            subset_j_data = subset_j_data.reshape((-1,) + subset_j_data.shape[1:3] + (1,))

        if model_type == "nn":
            subset_j_data = subset_j_data.astype('float32') / 255
            test_data = test_data.astype('float') / 255

        data_j = (subset_j_data, subset_j_targets, test_data, test_targets)
        # accuracy, loss, epochs, hist = build_model(dataset_type, model_type, data_j, n_epochs, lrate)
        accuracy = build_model(dataset_type, model_type, data_j, n_epochs, lrate)[0]

        accuracies.append(accuracy)

    return accuracies


# TODO: entry point 1
def run_multi_experiment(model_type, dataset_type, partitioning_type, n_clients, alpha, epochs, lrate=0.003):
    dict_accuracies = {}
    accuracies_mean = []
    accuracies_std = []

    for j, alpha_j in enumerate(alpha):
        accuracies_j = run_experiment(model_type, dataset_type, partitioning_type, n_clients, alpha_j, epochs, lrate)
        dict_accuracies.update({j: accuracies_j})
        mean_j = np.array(accuracies_j).mean()
        accuracies_mean.append(mean_j)
        std_j = np.array(accuracies_j).std()
        accuracies_std.append(std_j)

    plt.figure(figsize=(5, 3), dpi=300)
    plt.plot(np.arange(len(accuracies_mean)), accuracies_mean)
    plt.fill_between(np.arange(len(accuracies_mean)), np.array(accuracies_mean) - np.array(accuracies_std),
                     np.array(accuracies_mean) + np.array(accuracies_std), color='gray', alpha=0.2)
    plt.grid(linestyle='-', linewidth=0.5)
    # plt.suptitle(main_title, fontsize='small')
    plt.show()

    return dict_accuracies
