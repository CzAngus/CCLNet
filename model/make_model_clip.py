import torch
import torch.nn as nn
import numpy as np


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class visible_module(nn.Module):
    def __init__(self, model_name, h_resolution, w_resolution, vision_stride_size):
        super(visible_module, self).__init__()
        model_v = load_clip_to_cpu(model_name, h_resolution,w_resolution,vision_stride_size)
        # avg pooling to global pooling
        self.visible = model_v.visual
        # self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):

        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.conv2(x)
        x = self.visible.bn2(x)
        x = self.visible.relu(x)
        x = self.visible.conv3(x)
        x = self.visible.bn3(x)
        x = self.visible.relu(x)
        x = self.visible.avgpool(x)
        # x = self.pooling(x)
        return x

class thermal_module(nn.Module):
    def __init__(self, model_name, h_resolution, w_resolution, vision_stride_size):
        super(thermal_module, self).__init__()
        model_t = load_clip_to_cpu(model_name, h_resolution,w_resolution,vision_stride_size)
        # avg pooling to global pooling
        self.thermal = model_t.visual
        # self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):

        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.conv2(x)
        x = self.thermal.bn2(x)
        x = self.thermal.relu(x)
        x = self.thermal.conv3(x)
        x = self.thermal.bn3(x)
        x = self.thermal.relu(x)
        x = self.thermal.avgpool(x)
        # x = self.pooling(x)
        return x

class base_resnet(nn.Module):
    def __init__(self, model_name, h_resolution, w_resolution, vision_stride_size):
        super(base_resnet, self).__init__()
        model_base = load_clip_to_cpu(model_name, h_resolution, w_resolution, vision_stride_size)
        # avg pooling to global pooling
        # model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base.visual

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        # x_pool_att = self.base.attnpool(x)
        return x

class build_model(nn.Module):
    def __init__(self, args, num_classes_rgb, num_classes_ir):
        super(build_model, self).__init__()
        self.model_name = args.arch
        self.h_resolution = int((args.img_h-16) // args.stride_size[0] + 1)
        self.w_resolution = int((args.img_w-16) // args.stride_size[1] + 1)
        self.vision_stride_size = args.stride_size[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)

        self.thermal_module = thermal_module(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        self.visible_module = visible_module(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        self.base_resnet = base_resnet(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)

        self.num_classes_rgb = num_classes_rgb
        self.num_classes_ir = num_classes_ir
        self.prompt_learner = PromptLearner(self.num_classes_rgb, self.num_classes_ir, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)

        pool_dim = 2048
        self.gm_pool = 'on'
        self.num_features = pool_dim
        self.num_features_proj = 1024
        self.l2norm = Normalize(2)

        self.bottleneck = nn.BatchNorm1d(self.num_features)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1=None, x2=None, modal=0, get_image=False, get_text=False, label=None):
        if get_text:
            prompts = self.prompt_learner(label, modal=modal)
            if modal == 1:
                text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts_rgb)
            elif modal == 2:
                text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts_ir)
            else:
                return 0
            return text_features

        if get_image:
            if modal == 1:
                x = self.visible_module(x1)
            elif modal == 2:
                x = self.thermal_module(x2)
            else:
                return 0
            x = self.base_resnet(x)
            image_features_proj = self.base_resnet.base.attnpool(x)
            return image_features_proj[0]

        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        x = self.base_resnet(x)
        image_features_proj = self.base_resnet.base.attnpool(x)
        # image_features, _ = self.image_encoder(x)
        # x_pool = image_features_pool[0]

        # x = image_features
        if self.gm_pool  == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        feat = self.bottleneck(x_pool)

        if self.training:
            return feat, image_features_proj[0]
        else:
            return self.l2norm(feat)
            # return feat


class PromptLearner(nn.Module):
    def __init__(self, num_class_rgb, num_class_ir, dtype, token_embedding):
        super().__init__()

        ctx_init_rgb = "A visible photo of a X X X X person."
        ctx_init_ir = "An infrared photo of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init_rgb = ctx_init_rgb.replace("_", " ")
        ctx_init_ir = ctx_init_ir.replace("_", " ")
        n_ctx = 5

        tokenized_prompts_rgb = clip.tokenize(ctx_init_rgb)
        tokenized_prompts_ir = clip.tokenize(ctx_init_ir)
        with torch.no_grad():
            embedding_rgb = token_embedding(tokenized_prompts_rgb).type(dtype)
            embedding_ir = token_embedding(tokenized_prompts_ir).type(dtype)
        self.tokenized_prompts_rgb = tokenized_prompts_rgb  # torch.Tensor
        self.tokenized_prompts_ir = tokenized_prompts_ir  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors_rgb = torch.empty(num_class_rgb, n_cls_ctx, ctx_dim, dtype=dtype)
        cls_vectors_ir = torch.empty(num_class_ir, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors_rgb, std=0.02)
        nn.init.normal_(cls_vectors_ir, std=0.02)
        self.cls_ctx_rgb = nn.Parameter(cls_vectors_rgb)
        self.cls_ctx_ir = nn.Parameter(cls_vectors_ir)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix_rgb", embedding_rgb[:, :n_ctx + 1, :])
        self.register_buffer("token_prefix_ir", embedding_ir[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix_rgb", embedding_rgb[:, n_ctx + 1 + n_cls_ctx:, :])
        self.register_buffer("token_suffix_ir", embedding_ir[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class_rgb = num_class_rgb
        self.num_class_ir = num_class_ir
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label, modal=0):
        if modal == 1:
            cls_ctx_rgb = self.cls_ctx_rgb[label]
            b = label.shape[0]
            prefix_rgb = self.token_prefix_rgb.expand(b, -1, -1)
            suffix_rgb = self.token_suffix_rgb.expand(b, -1, -1)

            prompts = torch.cat(
                [
                    prefix_rgb,  # (n_cls, 1, dim)
                    cls_ctx_rgb,  # (n_cls, n_ctx, dim)
                    suffix_rgb,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            return  prompts
        elif modal == 2:
            cls_ctx_ir = self.cls_ctx_ir[label]
            b = label.shape[0]
            prefix_ir = self.token_prefix_ir.expand(b, -1, -1)
            suffix_ir = self.token_suffix_ir.expand(b, -1, -1)

            prompts = torch.cat(
                [
                    prefix_ir,  # (n_cls, 1, dim)
                    cls_ctx_ir,  # (n_cls, n_ctx, dim)
                    suffix_ir,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            return prompts

        else:
            prompts = None
            return 0




























