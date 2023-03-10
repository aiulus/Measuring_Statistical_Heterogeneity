import numpy as np
from sklearn.model_selection import StratifiedKFold
from utils import get_data

def partition_homo_skf(train, n_clients, alpha=0):
    subset_ID_map = {}
    x_train, y_train = train[0], train[1]
    # n_train = y_train.shape[0]
    skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=42)
    subsets = []
    for train_ID, test_ID in skf.split(train, y_train):
        subsets.append(test_ID)
    for j in range(n_clients):
        # np.random.shuffle(subsets[j])
        subset_ID_map[j] = subsets[j]

    return subset_ID_map


def partition_hetero_dir(train, n_clients, alpha):
    x_train, y_train = train[0], train[1]

    min_size = 0
    # classes
    K = len(y_train.unique())
    # data points
    N = y_train.shape[0]
    subset_ID_map = {}

    while min_size < 10:
        subset_ID_list = [[] for _ in range(n_clients)]
        for k in range(K):
            ids_k = np.where(y_train == k)[0]
            np.random.shuffle(ids_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
            proportions = np.array(
                [p * (len(ids_j) < N / n_clients) for p, ids_j in zip(proportions, subset_ID_list)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(ids_k)).astype(int)[:-1]
            subset_ID_list = [ids_j + ids.tolist() for ids_j, ids in
                              zip(subset_ID_list, np.split(ids_k, proportions))]
            min_size = min([len(ids_j) for ids_j in subset_ID_list])

    for j in range(n_clients):
        np.random.shuffle(subset_ID_list[j])
        subset_ID_map[j] = subset_ID_list[j]

    return subset_ID_map


def partition_quantity_based(train, n_clients, alpha):
    # sorted = pd.concat([y for x, y in data.groupby(0)]).reset_index().drop(columns=['index'])
    x, y = train[0], train[1]
    K = len(y.unique())
    # subsets_pure = {i: np.where(y == i)[0] for i in range(10)}
    subsets_pure = np.concatenate([np.where(y == i)[0] for i in range(K)])

    ids = np.arange(y.shape[0])
    batch_ids = np.array_split(ids, n_clients*alpha)
    minibatches = []

    for j in range(n_clients*alpha):
        batch_i = [subsets_pure[j] for j in batch_ids[j]]
        minibatches.append(batch_i)

    minibatches = np.array(minibatches)

    ids_mini = np.random.permutation(range(n_clients*alpha))
    subset_indices = np.array_split(ids_mini, n_clients)

    clients = {}

    for index in subset_indices:
        client_j = []
        for j in index:
            client_j.append(minibatches[j])
        client_j = np.concatenate(client_j)
        clients.update({j:np.array(client_j)})

    return clients


def partition(dataset_type, partitioning_type, n_clients, alpha):
    train, test = get_data(dataset_type, "D:/datasets")

    partition_funcs = {
        "homo": partition_homo_skf,
        "hetero-dir": partition_hetero_dir,
        "quant": partition_quantity_based
    }

    try:
        partition_func = partition_funcs[partitioning_type]
    except KeyError:
        raise ValueError(f"Invalid mode: {partitioning_type}")

    return partition_func(train, n_clients, alpha)