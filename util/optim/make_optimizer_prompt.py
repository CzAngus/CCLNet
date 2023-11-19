import torch


def make_optimizer(args, model_net):
    params = []
    for key, value in model_net.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = args.weight_decay
        if "bias" in key:
            lr = args.lr * args.bias_lr_factor
            weight_decay = args.weight_decay_bias

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer_net = getattr(torch.optim, 'Adam')(params)
    return optimizer_net




