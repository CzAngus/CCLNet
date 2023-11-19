import argparse
import time
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.backends import cudnn
from torch.autograd import Variable
from model.make_model_clip import build_model

from data.dataloader import TestData
from util.eval_metrics import eval_sysu, eval_regdb

from data.data_manager import process_query_sysu, process_gallery_sysu, process_test_regdb

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True

    main_worker(args)


def main_worker(args):

    ## load data
    print("==========\nargs:{}\n==========".format(args))
    data_set = args.dataset
    feat_dim=2048
    if data_set == 'sysu':
        data_path = '/home/cz/dataset/SYSU-MM01/'
        test_mode = [1, 2]
    elif data_set== 'regdb':
        data_path = '/home/cz/dataset/RegDB/'
        test_mode = [2, 1]

    ## build model
    print('==> Building model..')
    train_color_label = np.load(data_path +'pseudo_labels/' + 'train_rgb_resized_pseudo_label.npy')
    train_thermal_label = np.load(data_path +'pseudo_labels/' + 'train_ir_resized_pseudo_label.npy')
    if -1 in train_color_label:
        n_color_class = len(np.unique(train_color_label)) - 1
    else:
        n_color_class = len(np.unique(train_color_label)) 
    if -1 in train_thermal_label:
        n_thermal_class = len(np.unique(train_thermal_label)) - 1
    else:
        n_thermal_class = len(np.unique(train_thermal_label))

    model = build_model(args, n_color_class, n_thermal_class)
    model.cuda()


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalize,
    ])

    transform_visible = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalize,
    ])

    end = time.time()
    print('==> Loading data..')

    def extract_gall_feat(gall_loader):
        model.eval()
        print('Extracting Gallery Feature...')
        start = time.time()
        ptr = 0
        gall_feat = np.zeros((ngall, feat_dim))
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(gall_loader):
                batch_num = input.size(0)
                input = Variable(input.cuda())
                feat = model(input, input, modal=test_mode[0])
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                ptr = ptr + batch_num

        print("Extracting Time:\t {:.3f}".format(time.time() - start))
        return gall_feat

    def extract_query_feat(query_loader):
        model.eval()
        print ('Extracting Query Feature...')
        start = time.time()
        ptr = 0
        query_feat = np.zeros((nquery, feat_dim))
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(query_loader):
                batch_num = input.size(0)
                input = Variable(input.cuda())
                feat = model(input, input, modal=test_mode[1])
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                ptr = ptr + batch_num

        print('Extracting Time:\t {:.3f}'.format(time.time() - start))
        return query_feat

    if data_set == 'sysu':

        print('==> Resuming from checkpoint..')
        if os.path.isfile(args.resume_path):
            print('----load checkpoint-----')
            checkpoint = torch.load(args.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('no checkpoint found!')

        query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)
        nquery = len(query_label)
        ngall = len(gall_label)

        print("Dataset {} Statistics:".format(args.dataset))
        print("  ----------------------------")
        print("  subset   | # ids | # images")
        print("  ----------------------------")
        print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), len(query_label)))
        print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), len(gall_label)))
        print("  ----------------------------")

        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        query_loader = data.DataLoader(queryset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
        print("Data loading time:\t {:.3f}".format(time.time() - end))
        print('----------Testing----------')

        query_feat = extract_query_feat(query_loader)
        for trial in range(10):
            gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=trial)

            trial_gallset = TestData(gall_img, gall_label, transform=transform_visible, img_size=(args.img_w, args.img_h))
            trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

            gall_feat = extract_gall_feat(trial_gall_loader)
            distmat = np.matmul(query_feat, np.transpose(gall_feat))

            cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
            if trial == 0:
                all_cmc = cmc
                all_mAP = mAP
                all_mINP = mINP
            else:
                all_cmc = all_cmc + cmc
                all_mAP = all_mAP + mAP
                all_mINP = all_mINP + mINP

            print('Test Trial: {}'.format(trial))
            print("Performance: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}".format(cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
            print("-----------------------Next Trial--------------------")


    if data_set == 'regdb':
        for trial in range(10):
            test_trial = trial + 1
            print('==> Resuming from checkpoint..')
            model_path = args.resume_path
            if os.path.isfile(args.resume_path):
                print('----load checkpoint-----')
                checkpoint = torch.load(model_path)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                print('no checkpoint found!')

            query_img, query_label = process_test_regdb(data_path, trial=test_trial, modal='visible')
            gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modal='thermal')

            gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            gall_loader = data.DataLoader(gallset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

            nquery = len(query_label)
            ngall = len(gall_label)

            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
            print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

            query_feat = extract_query_feat(query_loader)
            gall_feat = extract_gall_feat(gall_loader)

            distmat = np.matmul(query_feat, np.transpose(gall_feat))
            cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)

            if trial == 0:
                all_cmc = cmc
                all_mAP = mAP
                all_mINP = mINP

            else:
                all_cmc = all_cmc + cmc
                all_mAP = all_mAP + mAP
                all_mINP = all_mINP + mINP

            print('Test Trial: {}'.format(trial))
            print(
                "Performance: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}".format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

    cmc = all_cmc / 10
    mAP = all_mAP / 10
    mINP = all_mINP / 10

    print("---------------ALl Performance---------------")
    print('All Average:')
    print(
        'Performance:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OTLA-ReID for testing")
    parser.add_argument('-d', '--dataset', type=str, default='sysu')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument("--resume", action="store_true", default=True, help="resume from checkpoint")
    parser.add_argument("--resume_path", type=str, metavar='PATH',
                        default="/home/cz/projects/checkpoint_mm/clippro/clip_pro_add.pth", help="checkpoint path")
    parser.add_argument('--seed', type=int, default=1)

    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'logs/'))
    # parser.add_argument('--resume_path', type=str, metavar='PATH', )

    parser.add_argument('-a', '--arch', type=str, default='RN50')
    parser.add_argument('--stride_size', type=list, default=[16,16])
    parser.add_argument('--pool-dim', type=int, default=2048)
    parser.add_argument('--per-add-iters',type=int,default=1)
    parser.add_argument('--img_w', default=144, type=int,
                        metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=288, type=int,
                        metavar='imgh', help='img height')
    parser.add_argument('--mode', default='all', type=str, help='all or indoor')
    parser.add_argument('--test-batch-size', default=64, type=int,
                        metavar='tb', help='testing batch size')

    args = parser.parse_args()

    main_worker(args)