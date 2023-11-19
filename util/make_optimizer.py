import torch


def make_optimizer_1stage(args, model):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "prompt_learner" in key:
            lr = args.stage1_baselr
            weight_decay = args.stage1_weight_decay
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]

    optimizer = getattr(torch.optim, 'Adam')(params)
    return optimizer


def make_optimizer_2stage(args, model_net):
    params = []
    keys = []
    for key, value in model_net.named_parameters():
        if "text_encoder" in key:
            value.requires_grad_(False)
            continue
        if "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
        lr = args.stage2_baselr
        weight_decay = args.stage2_weight_decay
        if "bias" in key:
            lr = args.stage2_baselr * args.stage2_bias_lr_factor
            weight_decay = args.stage2_weight_decay_bias

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]

    optimizer_net = getattr(torch.optim, 'Adam')(params)
    return optimizer_net

def make_optimizer_2stage_later(args, model_net):
    params = []
    keys = []
    for key, value in model_net.named_parameters():
        if "text_encoder" in key:
            value.requires_grad_(False)
            continue
        if "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
        lr = args.stage2_baselr * args.stage2_laterlr_factor
        weight_decay = args.stage2_weight_decay
        if "bias" in key:
            lr = args.stage2_baselr * args.stage2_bias_lr_factor
            weight_decay = args.stage2_weight_decay_bias

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]

    optimizer_net = getattr(torch.optim, 'Adam')(params)
    return optimizer_net


