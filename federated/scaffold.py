import torch


def initialize_c(model):
    return [torch.zeros_like(p.data) for p in model.parameters()]


def update_c(c_local, c_global, c_delta, weight):
    return [
        c_g + weight * (c_l - c_g + c_d)
        for c_g, c_l, c_d in zip(c_global, c_local, c_delta)
    ]


def compute_delta_c(c_local, c_global, lr, steps):
    return [(c_l - c_g) / (steps * lr) for c_l, c_g in zip(c_local, c_global)]
