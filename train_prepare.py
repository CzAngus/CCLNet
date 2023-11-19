import argparse
import os
import os.path as osp
from datetime import timedelta
import time
import sys
import random

import easydict
import numpy as np
import yaml

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.backends import cudnn

from train.train_0stage import do_train_stage0
from util.loss.make_loss import make_loss
from util.make_optimizer import make_optimizer_2stage
from util.optim.lr_scheduler import WarmupMultiStepLR
from util.optim.scheduler_factory import create_scheduler
from data.dataloader import Unlabeld_SYSUData
from model.make_model_clip import build_model
from util.utils import Logger

start_epoch = best_mAP = 0

def get_cluster_loader(dataset, batch_size, workers):
    cluster_loader = data.DataLoader(
        dataset,
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return cluster_loader


def main(args):
    # args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()
    cudnn.benchmark = True

    logs_time = args.logs_time
    logs_file = str(args.logs_file)

    sys.stdout = Logger(osp.join(args.logs_dir, logs_time, logs_file + '.txt'))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("==========\nArgs:{}\n==========".format(args))
    # Load datasets
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalizer,
    ])
    end = time.time()
    print("==> Load unlabeled dataset")
    data_path = args.data_path
    unlabel_dataset = Unlabeld_SYSUData(data_dir=data_path, transform=transform_test, rgb_cluster=False,
                                        ir_cluster=False)
    
    if -1 in unlabel_dataset.train_color_label:
        n_color_class = len(np.unique(unlabel_dataset.train_color_label)) - 1
    else:
        n_color_class = len(np.unique(unlabel_dataset.train_color_label)) 
    if -1 in unlabel_dataset.train_thermal_label:
        n_thermal_class = len(np.unique(unlabel_dataset.train_thermal_label)) - 1
    else:
        n_thermal_class = len(np.unique(unlabel_dataset.train_thermal_label))
    # num_classes = n_color_class + n_thermal_class

    print("Dataset {} Statistics:".format(args.dataset))
    print("  ----------------------------")
    print("  subset   | # ids | # images")
    print("  ----------------------------")
    print("  visible  | {:5d} | {:8d}".format(n_color_class, len(unlabel_dataset.train_color_image)))
    print("  thermal  | {:5d} | {:8d}".format(n_thermal_class, len(unlabel_dataset.train_thermal_image)))
    print("  ----------------------------")
    # print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), len(query_label)))
    # print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), len(gall_label)))
    print("  ----------------------------")
    print("Data loading time:\t {:.3f}".format(time.time() - end))

    # Create model
    model = build_model(args, n_color_class, n_thermal_class)
    model.cuda()

    optimizer_0stage = make_optimizer_2stage(args, model)
    scheduler_0stage = WarmupMultiStepLR(optimizer_0stage, args.stage2_steps, args.stage2_gamma, args.stage2_warmup_factor,
                                         args.stage2_warmup_iters, args.stage2_warmup_method)

    do_train_stage0(args, unlabel_dataset, model, optimizer_0stage, scheduler_0stage)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Contrastive learning on unsupervised Cross re-ID")
    args_main = parser.parse_args()

    args = yaml.load(open('config/config_sysu.yaml'), Loader=yaml.FullLoader)
    args = easydict.EasyDict(args)
    main(args)
