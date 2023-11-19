import time
import torch
import numpy as np
from torch.autograd import Variable
from .eval_metrics import eval_sysu, eval_regdb


def tester(args, epoch, main_net, test_mode, gall_label, gall_loader, query_label, query_loader, feat_dim=2048, query_cam=None, gall_cam=None, writer=None):
    # switch to evaluation mode
    main_net.eval()

    # print("Extracting Gallery Feature...")
    ngall = len(gall_label)
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = main_net(input, input, modal=test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    # print("Extracting Time:\t {:.3f}".format(time.time() - start))

    # print("Extracting Query Feature...")
    nquery = len(query_label)
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = main_net(input, input, modal=test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    # print("Extracting Time:\t {:.3f}".format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = -np.matmul(query_feat, np.transpose(gall_feat))
    # evaluation
    if args.dataset == "sysu":
        cmc, mAP, mINP = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
    elif args.dataset == "regdb":
        print("----------testing Regdb!")
        cmc, mAP, mINP = eval_regdb(distmat, query_label, gall_label)
    print("Evaluation Time:\t {:.3f}".format(time.time() - start))

    if writer is not None:
        writer.add_scalar("Rank1", cmc[0], epoch)
        writer.add_scalar("mAP", mAP, epoch)
        writer.add_scalar("mINP", mINP, epoch)

    return cmc, mAP, mINP