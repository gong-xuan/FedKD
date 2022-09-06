import pickle
import os 
import numpy as np 
import pandas as pd
import random
from PIL import Image
import cv2
import logging

import torch
from torch.utils.data import Dataset
from torchvision import transforms,datasets
from sklearn.model_selection import StratifiedShuffleSplit
import sys 
sys.path.append("..")
import utils.utils as utils
from torch.utils.data import DataLoader

class myImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

def dirichlet_datasplit(args, privtype='cifar10', publictype='cifar100', N_parties=20, online=True, public_percent=1):
    #public cifar100
    if publictype== 'cifar100':
        public_dataset = Cifar_Dataset( 
            os.path.join(args.datapath, 'cifar-100-python/'), publictype, train=True, verbose=False, distill = True, aug=online, public_percent=public_percent)
        # public_data = {}
        # public_data['x'] = public_dataset.img
        # import ipdb; ipdb.set_trace()
        # public_data['y'] = public_dataset.gt
        # public_data = public_dataset.img
    elif publictype== 'imagenet': #public_percent not valid
        public_dataset = myImageFolder(
            os.path.join(args.datapath, 'imagenet/train/'),
            transforms.Compose([
                transforms.RandomResizedCrop(32), #224
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        )
        # public_data = public_dataset.imgs
    # import ipdb; ipdb.set_trace()
    distill_loader = DataLoader(
            dataset=public_dataset, batch_size=args.disbatchsize, shuffle=online, 
            num_workers=args.num_workers, pin_memory=True, sampler=None)
    #private
    if privtype=='cifar10':
        subpath = 'cifar-10-batches-py/'
        N_class = 10
    elif privtype=='cifar100':
        subpath = 'cifar-100-python/'
        N_class = 100
    
    splitname = f'./splitfile/{privtype}/{args.alpha}_{args.seed}.npy'
    if os.path.exists(splitname):
        split_arr =  np.load(splitname)
        assert split_arr.shape == (N_class, N_parties)
    else:
        split_arr = np.random.dirichlet([args.alpha]*N_parties, N_class)#nclass*N_parties
        np.save(splitname, split_arr)
    
    test_dataset = Cifar_Dataset(
        os.path.join(args.datapath, subpath), privtype, train=False, verbose=False)
    train_dataset = Cifar_Dataset( 
        os.path.join(args.datapath, subpath), privtype, train=True, verbose=False)
    train_x, train_y = train_dataset.img, train_dataset.gt
    priv_data = [None] * N_parties
    for cls_idx in range(N_class):
        idx = np.where(train_y == cls_idx)[0]
        totaln = idx.shape[0]
        idx_start = 0
        for i in range(N_parties):
            if i==N_parties-1:
                cur_idx = idx[idx_start:]
            else:
                idx_end = idx_start + int(split_arr[cls_idx][i]*totaln)
                cur_idx = idx[idx_start: idx_end]
                idx_start = idx_end
            if cur_idx == ():
                continue
            if priv_data[i] is None:
                priv_data[i] = {}
                priv_data[i]['x'] = train_x[cur_idx]
                priv_data[i]['y'] = train_y[cur_idx]
                priv_data[i]['idx'] = cur_idx
            else:
                priv_data[i]['idx'] = np.r_[(priv_data[i]['idx'], cur_idx)]
                priv_data[i]['x'] = np.r_[(priv_data[i]['x'], train_x[cur_idx])]
                priv_data[i]['y'] = np.r_[(priv_data[i]['y'], train_y[cur_idx])]
    all_priv_data = {}
    all_priv_data['x'] = train_x
    all_priv_data['y'] = train_y
    return priv_data, train_dataset, test_dataset, public_dataset, distill_loader

class Cifar_Dataset:
    def __init__(self, local_dir, data_type, train=True, with_coarse_label=False, verbose=False, distill=False, aug=True, public_percent=1):
        self.distill = distill
        if data_type == 'cifar10':
            if train == True:
                img, gt = [], []
                for i in range(1, 6):
                    file_name = None
                    file_name = os.path.join(local_dir + 'data_batch_{0}'.format(i))
                    X_tmp, y_tmp = (None, None)
                    with open(file_name, 'rb') as (fo):
                        datadict = pickle.load(fo, encoding='bytes')
                    X_tmp = datadict[b'data']
                    y_tmp = datadict[b'labels']
                    X_tmp = X_tmp.reshape(10000, 3, 32, 32)
                    y_tmp = np.array(y_tmp)
                    img.append(X_tmp)
                    gt.append(y_tmp)
                img = np.vstack(img)
                gt = np.hstack(gt)
            else:
                file_name = None
                file_name = os.path.join(local_dir + 'test_batch')
                with open(file_name, 'rb') as (fo):
                    datadict = pickle.load(fo, encoding='bytes')
                    img = datadict[b'data']
                    gt = datadict[b'labels']
                    img = img.reshape(10000, 3, 32, 32)
                    gt = np.array(gt)
        elif data_type == 'cifar100':
            if train == True:
                file_name = None
                file_name = os.path.abspath(local_dir + 'train')
                with open(file_name, 'rb') as (fo):
                    datadict = pickle.load(fo, encoding='bytes')
                    img = datadict[b'data']
                    if with_coarse_label:
                        gt = datadict[b'coarse_labels']
                    else:
                        gt = datadict[b'fine_labels']
                    img = img.reshape(50000, 3, 32, 32)
                    gt = np.array(gt)
            else:
                file_name = None
                file_name = os.path.join(local_dir + 'test')
                with open(file_name, 'rb') as (fo):
                    datadict = pickle.load(fo, encoding='bytes')
                    img = datadict[b'data']
                    if with_coarse_label:
                        gt = datadict[b'coarse_labels']
                    else:
                        gt = datadict[b'fine_labels']
                    # import ipdb; ipdb.set_trace()
                    img = img.reshape(10000, 3, 32, 32)
                    gt = np.array(gt)
        else:
            logging.info('Unknown Data type. Stopped!')
            return
        if verbose:
            logging.info(f'img shape: {img.shape}')
            logging.info(f'label shape: {gt.shape}')
        self.img = np.asarray(img)
        self.img = self.img.transpose((0, 2, 3, 1))
        self.gt = np.asarray(gt)
        total_N_img = img.shape[0]
        # import ipdb; ipdb.set_trace()
        if public_percent<1:
            total_N_img = int(total_N_img*public_percent)
            self.img = self.img[:total_N_img]
            self.gt = self.gt[:total_N_img]
            logging.info(f'Clip with {public_percent}, to {total_N_img}')
        self.fixid = np.arange(total_N_img)
        self.aug = aug
        self.train = train
        
        self.train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            utils.Cutout(16),
            ])
        
        self.test_transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))])

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, idx):
        image = self.img[idx]
        label = self.gt[idx]
        fixid = self.fixid[idx]
        # transimage = Image.fromarray(image.transpose(1,2,0).astype('uint8'))
        # transimage = transforms.ToPILImage()(transimage)
        transimage = Image.fromarray(image)
        
        if self.train and self.aug:
            transformer = self.train_transformer 
        else:
            transformer = self.test_transformer
        transimage = transformer(transimage)
        if self.distill:
            return (transimage, label, idx)
        else:
            return (transimage, label, fixid)


class Dataset_fromarray(Cifar_Dataset):
    def __init__(self, img_array, gt_array, train=True, verbose=False, multitrans=1, distill=False, aug=True):
        self.distill = distill
        self.img = img_array
        self.gt = gt_array
        self.fixid = np.arange(self.img.shape[0])
        self.multitrans = multitrans
        self.train = train
        
        self.train_transformer = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                utils.Cutout(16),
                ])
        
        self.test_transformer = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))])
        # self.transformer2= transforms.Compose(self.transformer.transforms[:-1]) #if multi, no cutout
        if verbose == True:
            logging.info(f'img shape: {self.img.shape}')
            logging.info(f'label shape: {self.gt.shape}')
        self.aug = aug

def generate_alignment_data(data_set, N_alignment=3000):
    X, y = data_set['x'], data_set['y']
    split = StratifiedShuffleSplit(n_splits=1, train_size=N_alignment)
    if N_alignment == X.shape[0]:
        alignment_data = {}
        alignment_data['idx'] = np.arange(y.shape[0])
        alignment_data['x'] = X
        alignment_data['y'] = y
        return alignment_data
    else:
        for train_index, _ in split.split(X, y):
            X_alignment = X[train_index]
            y_alignment = y[train_index]

        alignment_data = {}
        alignment_data['idx'] = train_index
        alignment_data['x'] = X_alignment
        alignment_data['y'] = y_alignment
        return alignment_data


def cifar_fd_data(train_data, N_class, N_parties, N_samples_per_class):
    train_x, train_y = train_data.img, train_data.gt
    priv_data = [None] * N_parties
    all_priv_idx = np.array([], dtype=(np.int16))
    all_publ_idx = np.array([], dtype=(np.int16))
    private_n = N_parties * N_samples_per_class
    for cls_idx in range(N_class):
        idx = np.where(train_y == cls_idx)[0]
        priv_idx = idx[:private_n]
        all_priv_idx = np.r_[(all_priv_idx, priv_idx)]
        public_idx = idx[private_n:] #rest
        all_publ_idx = np.r_[(all_publ_idx, public_idx)]
        for i in range(N_parties):
            idx_tmp = priv_idx[i * N_samples_per_class:(i + 1) * N_samples_per_class]
            if priv_data[i] is None:
                tmp = {}
                tmp['x'] = train_x[idx_tmp]
                tmp['y'] = train_y[idx_tmp]
                tmp['idx'] = idx_tmp
                priv_data[i] = tmp
            else:
                priv_data[i]['idx'] = np.r_[(priv_data[i]['idx'], idx_tmp)]
                priv_data[i]['x'] = np.r_[(priv_data[i]['x'], train_x[idx_tmp])]
                priv_data[i]['y'] = np.r_[(priv_data[i]['y'], train_y[idx_tmp])]

    total_priv_data = {}
    total_priv_data['idx'] = all_priv_idx
    total_priv_data['x'] = train_x[all_priv_idx]
    total_priv_data['y'] = train_y[all_priv_idx]
    public_data = {}
    public_data['idx'] = all_publ_idx
    public_data['x'] = train_x[all_publ_idx]
    public_data['y'] = train_y[all_publ_idx]
    return (priv_data, total_priv_data, public_data)


