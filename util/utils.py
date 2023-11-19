import copy
import os
import random
from collections import defaultdict
import torch.distributed as dist

import numpy as np
from torch.utils.data.sampler import Sampler
import sys
import os.path as osp
import torch
import errno

def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of color image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label
    

def GenIdx( train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k,v in enumerate(train_color_label) if v==unique_label_color[i]]
        color_pos.append(tmp_pos)
        
    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k,v in enumerate(train_thermal_label) if v==unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos
    
def GenIdx_single(label):
    pos = []
    num = []
    unique_label = np.unique(label)
    for i in range(np.max(unique_label)+1):
        if i in unique_label:
            tmp_pos = [k for k, v in enumerate(label) if v == i]
            pos.append(tmp_pos)
            num.append(len(tmp_pos))
        else:
            pos.append([])
            num.append(0)

    return pos, np.array(num) / np.array(num).sum()

def GenCamIdx(gall_img, gall_label, mode):
    if mode =='indoor':
        camIdx = [1,2]
    else:
        camIdx = [1,2,4,5]
    gall_cam = []
    for i in range(len(gall_img)):
        gall_cam.append(int(gall_img[i][-10]))
    
    sample_pos = []
    unique_label = np.unique(gall_label)
    for i in range(len(unique_label)):
        for j in range(len(camIdx)):
            id_pos = [k for k,v in enumerate(gall_label) if v==unique_label[i] and gall_cam[k]==camIdx[j]]
            if id_pos:
                sample_pos.append(id_pos)
    return sample_pos
    

class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, color_pos, num_pos, batchSize):
        uni_label_color = np.unique(train_color_label)
        self.n_classes = len(uni_label_color)
        print(self.n_classes)

        N = len(train_color_label)
        for j in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx = np.random.choice(uni_label_color, batchSize, replace=False)

            for i in range(batchSize):
                sample_color = np.random.choice(color_pos[batch_idx[i]], num_pos)

                if j == 0 and i == 0:
                    index = sample_color
                else:
                    index = np.hstack((index, sample_color))

        self.index = index
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index)))

    def __len__(self):
        return self.N

class IdentitySampler_nosk(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize):
        uni_label_color = np.unique(train_color_label)
        # uni_label_color = np.delete(uni_label_color,index=0)
        uni_label_thermal = np.unique(train_thermal_label)
        # uni_label_thermal = np.delete(uni_label_thermal, index=0)
        # self.n_classes = len(uni_label_color)
        print("len of uni_label_color:", len(uni_label_color))
        print("len of uni_label_thermal:", len(uni_label_thermal))
        print('IdentitySampler_nosk----')

        N = np.maximum(len(train_color_label), len(train_thermal_label)) 
        for j in range(int(N/(batchSize*num_pos))+1):
            batch_idx_rgb = np.random.choice(uni_label_color, batchSize, replace=False)
            batch_idx_ir = np.random.choice(uni_label_thermal, batchSize, replace=False)
            for i in range(batchSize):
                sample_color  = np.random.choice(color_pos[batch_idx_rgb[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx_ir[i]], num_pos)
                
                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1
        self.index2 = index2
        self.N  = N
        
    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N


class IdentitySampler_nosk_all(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize):
        uni_label_color = np.unique(train_color_label)
        uni_label_color = torch.randperm(len(uni_label_color)).tolist()

        uni_label_thermal = np.unique(train_thermal_label)
        uni_label_thermal = torch.randperm(len(uni_label_thermal)).tolist()

        print("len of uni_label_color:", len(uni_label_color))
        print("len of uni_label_thermal:", len(uni_label_thermal))

        N = np.minimum(len(uni_label_color), len(uni_label_thermal))

        for j in range(N):
            sample_color = np.random.choice(color_pos[uni_label_color[j]], num_pos)
            sample_thermal = np.random.choice(thermal_pos[uni_label_thermal[j]], num_pos)

            if j == 0:
                index1 = sample_color
                index2 = sample_thermal
            else:
                index1 = np.hstack((index1, sample_color))
                index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N


class IdentitySampler_nosk_unique(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
            为了每一个人都被尽量的采样到而设置的采样方式
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize):
        uni_label_color = np.unique(train_color_label)
        uni_label_thermal = np.unique(train_thermal_label)
        N = np.maximum(len(train_color_label), len(train_thermal_label))

        uni_label_color_temp = uni_label_color
        uni_label_thermal_temp = uni_label_thermal

        batch_idx_rgb_list = []
        batch_idx_ir_list = []
        for j in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx_rgb = []
            batch_idx_ir = []
            for i in range(batchSize):
                if len(uni_label_color_temp) == 0:
                    uni_label_color_temp = uni_label_color
                if len(uni_label_thermal_temp) == 0:
                    uni_label_thermal_temp = uni_label_thermal

                idx_rgb = random.randint(0, len(uni_label_color_temp) - 1)
                idx_ir = random.randint(0, len(uni_label_thermal_temp) - 1)
                batch_idx_rgb.append(uni_label_color_temp[idx_rgb])
                batch_idx_ir.append(uni_label_thermal_temp[idx_ir])
                uni_label_color_temp = np.delete(uni_label_color_temp, idx_rgb)
                uni_label_thermal_temp = np.delete(uni_label_thermal_temp, idx_ir)
            batch_idx_rgb_list.append(np.array(batch_idx_rgb))
            batch_idx_ir_list.append(np.array(batch_idx_ir))

        for j in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx_rgb = batch_idx_rgb_list[j]
            batch_idx_ir = batch_idx_ir_list[j]

            for i in range(batchSize):
                sample_color = np.random.choice(color_pos[batch_idx_rgb[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx_ir[i]], num_pos)

                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 
def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """  
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
            
def set_seed(seed, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def set_requires_grad(nets, requires_grad=False):
            """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
            Parameters:
                nets (network list)   -- a list of networks
                requires_grad (bool)  -- whether the networks require gradients or not
            """
            if not isinstance(nets, list):
                nets = [nets]
            for net in nets:
                if net is not None:
                    for param in net.parameters():
                        param.requires_grad = requires_grad
