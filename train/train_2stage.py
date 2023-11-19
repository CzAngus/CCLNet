import argparse
import collections
import os
import os.path as osp
import shutil
from datetime import timedelta
import time
import sys
import random

import easydict
import numpy as np
import yaml
from sklearn.cluster import DBSCAN

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
# from tensorboardX import SummaryWriter
from torch.cuda import amp

from util.eval_metrics import extract_features_clip
from util.faiss_rerank import compute_jaccard_distance
from util.loss.supcontrast import SupConLoss
from util.utils import AverageMeter

from ClusterContrast.cm import ClusterMemory
from data.data_manager import process_query_sysu, process_gallery_sysu
from data.dataloader import SYSUData_Stage2, IterLoader, TestData
from util.eval import tester
from util.utils import IdentitySampler_nosk, GenIdx

from util.make_optimizer import make_optimizer_2stage, make_optimizer_2stage_later
from util.optim.lr_scheduler import WarmupMultiStepLR

# torch.backends.cuda.max_split_size_mb = 2750

def get_cluster_loader(dataset, batch_size, workers):
    cluster_loader = data.DataLoader(
        dataset,
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return cluster_loader


def do_train_stage2(args,
                    unlabel_dataset,
                    model,
                    optimizer,
                    scheduler,
                    loss_fn_rgb,
                    loss_fn_ir,
                    ):
    best_acc = 0
    device = 'cuda'
    epochs = args.stage2_maxepochs
    start_time = time.monotonic()

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomCrop((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalizer,
        transforms.RandomErasing(p=0.5)
    ])
    transform_train_ir = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomCrop((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalizer,
        transforms.RandomErasing(p=0.5),
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalizer,
    ])


    batch = args.stage2_ims_per_batch
    num_classes_rgb = model.num_classes_rgb
    num_classes_ir = model.num_classes_ir
    i_ter_rgb = num_classes_rgb // batch
    i_ter_ir = num_classes_ir // batch
    left_rgb = num_classes_rgb-batch* (num_classes_rgb//batch)
    left_ir = num_classes_ir-batch* (num_classes_ir//batch)
    if left_rgb != 0 :
        i_ter_rgb = i_ter_rgb+1
    if left_ir != 0 :
        i_ter_ir = i_ter_ir+1
    text_features_rgb = []
    text_features_ir = []
    with torch.no_grad():
        for i in range(i_ter_rgb):
            if i+1 != i_ter_rgb:
                l_list_rgb = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list_rgb = torch.arange(i*batch, num_classes_rgb)
            # with amp.autocast(enabled=True):
            text_feature_rgb = model(get_text = True, label = l_list_rgb, modal=1)
            text_features_rgb.append(text_feature_rgb.cpu())
        text_features_rgb = torch.cat(text_features_rgb, 0).cuda()
    with torch.no_grad():
        for i in range(i_ter_ir):
            if i+1 != i_ter_ir:
                l_list_ir = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list_ir = torch.arange(i*batch, num_classes_ir)
            # with amp.autocast(enabled=True):
            text_feature_ir = model(get_text = True, label = l_list_ir, modal=2)
            text_features_ir.append(text_feature_ir.cpu())
        text_features_ir = torch.cat(text_features_ir, 0).cuda()

    scaler = amp.GradScaler()
    losses = AverageMeter()
    losses_rgb = AverageMeter()
    losses_ir = AverageMeter()
    losses_i2t = AverageMeter()
    losses_i2t_rgb = AverageMeter()
    losses_i2t_ir = AverageMeter()

    # torch.cuda.empty_cache()

    for epoch in range(1, epochs+1):
        with torch.no_grad():
            if epoch == 1:
                # DBSCAN cluster
                eps = args.eps
                print('Clustering criterion: eps: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled data')
            print("==> Extract RGB features")
            unlabel_dataset.rgb_cluster = True
            unlabel_dataset.ir_cluster = False
            cluster_loader_rgb = get_cluster_loader(unlabel_dataset, args.test_batch_size, args.workers)
            features_rgb, _ = extract_features_clip(model, cluster_loader_rgb, modal=1, get_image=False)
            features_rgb = torch.cat([features_rgb[path].unsqueeze(0) for path in unlabel_dataset.train_color_path], 0).cuda()

            print("==> Extract IR features")
            unlabel_dataset.ir_cluster = True
            unlabel_dataset.rgb_cluster = False
            cluster_loader_ir = get_cluster_loader(unlabel_dataset, args.test_batch_size, args.workers)
            features_ir, _ = extract_features_clip(model, cluster_loader_ir, modal=2, get_image=False)
            features_ir = torch.cat([features_ir[path].unsqueeze(0) for path in unlabel_dataset.train_thermal_path], 0).cuda()

            rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2)
            pseudo_labels_rgb = cluster.fit_predict(rerank_dist_rgb)
            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)

            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2)
            pseudo_labels_ir = cluster.fit_predict(rerank_dist_ir)
            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)

        del features_rgb, features_ir, cluster_loader_rgb, cluster_loader_ir, rerank_dist_rgb, rerank_dist_ir

        if epoch >= args.base_epoch:
            change_scale = args.change_scale
            print('----------Start Memory(rgb and ir) Change!----------')
        else:
            change_scale = 1.


        memory = ClusterMemory(2048, num_cluster_rgb, num_cluster_ir, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard, change_scale=change_scale).cuda()
        memory.features_rgb = F.normalize(cluster_features_rgb, dim=1).cuda()
        memory.features_ir = F.normalize(cluster_features_ir, dim=1).cuda()

        # generate new dataset
        end = time.time()
        trainset = SYSUData_Stage2(args.data_path, pseudo_labels_rgb, pseudo_labels_ir, transform_train_rgb,
                                 transform_train_ir)
        print("New Dataset Information---- ")
        print("  ----------------------------")
        print("  subset   | # ids | # images")
        print("  ----------------------------")
        print("  visible  | {:5d} | {:8d}".format(len(np.unique(trainset.train_color_pseudo_label)),
                                                  len(trainset.train_color_image)))
        print("  thermal  | {:5d} | {:8d}".format(len(np.unique(trainset.train_thermal_pseudo_label)),
                                                  len(trainset.train_thermal_image)))
        print("  ----------------------------")
        print("Data loading time:\t {:.3f}".format(time.time() - end))

        color_pos, thermal_pos = GenIdx(trainset.train_color_pseudo_label, trainset.train_thermal_pseudo_label)

        sampler = IdentitySampler_nosk(trainset.train_color_pseudo_label, trainset.train_thermal_pseudo_label, color_pos, thermal_pos,
                                       args.num_instances, args.batch_size)

        trainset.cIndex = sampler.index1
        trainset.tIndex = sampler.index2

        trainloader = data.DataLoader(trainset, batch_size=args.batch_size * args.num_instances, sampler=sampler,
                                      num_workers=args.workers,
                                      drop_last=True)

        losses.reset()
        losses_rgb.reset()
        losses_ir.reset()
        losses_i2t.reset()
        losses_i2t_rgb.reset()
        losses_i2t_ir.reset()

        scheduler.step()
        model.train()
        for n_iter, (img_rgb, img_ir, label_rgb, label_ir, vid_rgb, vid_ir) in enumerate(trainloader):

            optimizer.zero_grad()
            img_rgb = img_rgb.to(device)
            label_rgb = label_rgb.to(device)
            vid_rgb = vid_rgb.to(device)

            img_ir = img_ir.to(device)
            label_ir = label_ir.to(device)
            vid_ir = vid_ir.to(device)

            # with amp.autocast(enabled=True):
            image_features, image_features_proj = model(x1=img_rgb, x2=img_ir, modal=0)
            logits_rgb = image_features_proj[:img_rgb.size(0)] @ text_features_rgb.t()
            logits_ir = image_features_proj[img_rgb.size(0):] @ text_features_ir.t()
            loss_i2t_rgb = loss_fn_rgb(logits_rgb, vid_rgb)
            loss_i2t_ir = loss_fn_ir(logits_ir, vid_ir)
            loss_i2t = loss_i2t_rgb + loss_i2t_ir

            # out_rgb = image_features[:img_rgb.size(0)]
            out_rgb = image_features[:img_rgb.size(0)]
            out_ir = image_features[img_rgb.size(0):]
            loss_rgb, loss_ir = memory(out_rgb, out_ir, label_rgb, label_ir)

            loss = loss_rgb + loss_ir + loss_i2t

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses_rgb.update(loss_rgb.item())
            losses_ir.update(loss_ir.item())
            losses_i2t.update(loss_i2t.item())
            losses_i2t_rgb.update(loss_i2t_rgb.item())
            losses_i2t_ir.update(loss_i2t_ir.item())
            losses.update(loss.item())
            torch.cuda.synchronize()
            if n_iter % 100 == 0:
                print("Epoch[{}] Iteration[{}/{}], Loss_rgb_ir_i2t: ({:.3f})({:.3f})({:.3f})({:.3f}) ({:.3f}), Base Lr: {:.2e}"
                 .format(epoch, (n_iter + 1), len(trainloader), losses_rgb.avg, losses_ir.avg,
                         losses_i2t_rgb.avg, losses_i2t_ir.avg, losses.avg,scheduler.get_lr()[0]))

        if epoch % args.eval_step == 0 or (epoch == args.stage2_maxepochs):
            print('Test Epoch: {}'.format(epoch))
            test_mode = [1, 2]
            query_img, query_label, query_cam = process_query_sysu(args.data_path, mode=args.mode)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

            for trial in range(10):
                # print('-------test trial {}-------'.format(trial))
                gall_img, gall_label, gall_cam = process_gallery_sysu(args.data_path, mode=args.mode, trial=trial)
                gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                gall_loader = data.DataLoader(gallset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

                cmc, mAP, mINP = tester(args, epoch, model, test_mode, gall_label, gall_loader, query_label, query_loader,
                                        feat_dim=2048,
                                        query_cam=query_cam, gall_cam=gall_cam)

                if trial == 0:
                    all_cmc = cmc
                    all_mAP = mAP
                    all_mINP = mINP
                else:
                    all_cmc = all_cmc + cmc
                    all_mAP = all_mAP + mAP
                    all_mINP = all_mINP + mINP

            cmc = all_cmc / 10
            mAP = all_mAP / 10
            mINP = all_mINP / 10
            print(
                "Performance[ALL]: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}".format(
                    cmc[0], cmc[4],
                    cmc[9], cmc[19],
                    mAP, mINP))

            if cmc[0] > best_acc:
                best_acc = cmc[0]
                best_epoch = epoch
                best_mAP = mAP
                best_mINP = mINP
                state = {
                    "state_dict": model.state_dict(),
                    "cmc": cmc,
                    "mAP": mAP,
                    "mINP": mINP,
                    "epoch": epoch,
                }
                torch.save(state, os.path.join(args.model_path, args.logs_file + "_stage2.pth"))
            print("Best Epoch [{}], Rank-1: {:.2%} |  mAP: {:.2%}| mINP: {:.2%}".format(best_epoch, best_acc, best_mAP, best_mINP))

        torch.cuda.empty_cache()

    end_time = time.monotonic()
    print('Stage2 running time: ', timedelta(seconds=end_time - start_time))

