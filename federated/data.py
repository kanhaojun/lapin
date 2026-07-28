import copy

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


def average_weights(weights):
    w_avg = copy.deepcopy(weights[0])
    for key in w_avg:
        for i in range(1, len(weights)):
            w_avg[key] += weights[i][key]
        w_avg[key] = torch.div(w_avg[key], len(weights))
    return w_avg


def data_iid(dataset, num_users, verbose=False):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, list(range(len(dataset)))
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        if verbose:
            print(f'client {i}: {len(dict_users[i])} samples')
    return dict_users


def data_noniid(dataset, num_users, verbose=False):
    num_items = len(dataset)
    base_num_items_per_user = num_items // num_users
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    all_idxs = np.arange(num_items)

    image_values = []
    for i in range(num_items):
        image, _ = dataset[i]
        if isinstance(image, torch.Tensor):
            image_values.append(torch.mean(image).item())
        else:
            image_values.append(np.mean(image))

    sorted_idxs = all_idxs[np.argsort(image_values)]
    remaining_idxs = set(sorted_idxs)
    for i in range(num_users):
        num_items_for_user = np.random.randint(
            int(base_num_items_per_user * 0.8),
            int(base_num_items_per_user * 1.2),
        )
        num_items_for_user = min(num_items_for_user, len(remaining_idxs))
        user_data = np.random.choice(list(remaining_idxs), num_items_for_user, replace=False)
        dict_users[i] = user_data
        remaining_idxs -= set(user_data)
        if verbose:
            print(f'client {i}: {len(dict_users[i])} samples')
    return dict_users


def build_client_dataloaders(
    datasets,
    num_clients,
    batch_size,
    num_workers,
    iid=False,
    verbose=False,
):
    if len(datasets) == 1:
        dataset = datasets[0]
        split_fn = data_iid if iid else data_noniid
        dict_users = split_fn(dataset, num_clients, verbose)
        client_splits = [
            DatasetSplit(dataset, dict_users[i]) for i in range(num_clients)
        ]
    else:
        split_fn = data_iid if iid else data_noniid
        dict_users_list = [
            split_fn(dataset, num_clients, verbose) for dataset in datasets
        ]
        client_splits = []
        for i in range(num_clients):
            combined = ConcatDataset([
                DatasetSplit(dataset, dict_users_list[j][i])
                for j, dataset in enumerate(datasets)
            ])
            client_splits.append(combined)

    return [
        DataLoader(
            split,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )
        for split in client_splits
    ]
