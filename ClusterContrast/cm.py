from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
from scipy.optimize import linear_sum_assignment


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs_rgb, inputs_ir, targets_rgb, targets_ir, features_rgb, features_ir, momentum, scale):
        ctx.features_rgb = features_rgb
        ctx.features_ir = features_ir
        ctx.momentum = momentum
        ctx.scale = scale

        ctx.save_for_backward(inputs_rgb, inputs_ir, targets_rgb, targets_ir)
        outputs_rgb = inputs_rgb.mm(ctx.features_rgb.t())
        outputs_ir = inputs_ir.mm(ctx.features_ir.t())

        return outputs_rgb, outputs_ir

    @staticmethod
    def backward(ctx, grad_outputs1, grad_outputs2):
        inputs_rgb, inputs_ir, targets_rgb, targets_ir = ctx.saved_tensors
        grad_inputs1 = None
        grad_inputs2 = None
        if ctx.needs_input_grad[0]:
            grad_inputs1 = grad_outputs1.mm(ctx.features_rgb)
        if ctx.needs_input_grad[1]:
            grad_inputs2 = grad_outputs2.mm(ctx.features_ir)

        # momentum update
        for x,y in zip(inputs_rgb, targets_rgb):
            ctx.features_rgb[y] = ctx.momentum * ctx.features_rgb[y] + (1. - ctx.momentum) * x
            ctx.features_rgb[y] /= ctx.features_rgb[y].norm()

        for x, y in zip(inputs_ir, targets_ir):
            ctx.features_ir[y] = ctx.momentum * ctx.features_ir[y] + (1. - ctx.momentum) * x
            ctx.features_ir[y] /= ctx.features_ir[y].norm()

        if ctx.scale != 1.:
            # Hungarian Matching Phase
            cost_matrix = ctx.features_rgb.mm(ctx.features_ir.t())
            rgb_idx, ir_idx = linear_sum_assignment(cost_matrix.detach().cpu().numpy(), maximize=True)

            temp_rgb = ctx.features_rgb
            for r_idx, i_idx in zip(rgb_idx,ir_idx):
                if cost_matrix[r_idx, i_idx] >= 0:
                    if r_idx in targets_rgb:
                        ctx.features_rgb[r_idx] = ctx.scale * ctx.features_rgb[r_idx] + (1. - ctx.scale) * ctx.features_ir[i_idx]
                        ctx.features_rgb[r_idx] /= ctx.features_rgb[r_idx].norm()
                    if i_idx in targets_ir:
                        ctx.features_ir[i_idx] =  ctx.scale * ctx.features_ir[i_idx] + (1. - ctx.scale) * temp_rgb[r_idx]
                        ctx.features_ir[i_idx] /= ctx.features_ir[i_idx].norm()

            del temp_rgb

        return grad_inputs1, grad_inputs2, None, None, None, None, None, None


# outputs = cm(inputs, targets, rgb_size, self.features, self.num_samples_rgb, self.num_samples_ir, self.momentum)
def cm(inputs_rgb, inputs_ir, indexes_rgb, indexes_ir, features_rgb, features_ir, momentum, change_scale):
    return CM.apply(inputs_rgb, inputs_ir, indexes_rgb ,indexes_ir, features_rgb, features_ir, torch.Tensor([momentum]).to(inputs_rgb.device), torch.Tensor([change_scale]).to(inputs_rgb.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples_rgb, num_samples_ir, temp=0.05, momentum=0.9, use_hard=False, change_scale=0.9):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples_rgb = num_samples_rgb
        self.num_samples_ir = num_samples_ir

        self.momentum = momentum
        self.change_scale = change_scale
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features_rgb', torch.zeros(num_samples_rgb, num_features))
        self.register_buffer('features_ir', torch.zeros(num_samples_ir, num_features))


    def forward(self, inputs_rgb, inputs_ir, targets_rgb, targets_ir):
        inputs_rgb = F.normalize(inputs_rgb, dim=1).cuda()
        inputs_ir = F.normalize(inputs_ir, dim=1).cuda()

        outputs_rgb, outputs_ir = cm(inputs_rgb, inputs_ir, targets_rgb, targets_ir, self.features_rgb, self.features_ir, self.momentum, self.change_scale)
        outputs_rgb /= self.temp
        outputs_ir /= self.temp
        loss_rgb = F.cross_entropy(outputs_rgb, targets_rgb)
        loss_ir = F.cross_entropy(outputs_ir, targets_ir)

        return loss_rgb, loss_ir


