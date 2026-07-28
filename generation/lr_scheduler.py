def lr_lambda(step):
    warm_up_steps = 20
    lr_max = 0.01
    lr_min = 3e-4
    if step <= warm_up_steps:
        lr = max((step / warm_up_steps) * lr_max, lr_min)
    else:
        lr = max(lr_min, lr_max * 0.9 ** (step - warm_up_steps))
    return lr / lr_min
