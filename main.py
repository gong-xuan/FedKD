import argparse
import os
from tensorboardX import SummaryWriter
import logging
from datetime import datetime
import torch 
# import sys 
# sys.path.append("..")
import models.resnet8 as resnet8
import dataset.data_cifar as data_cifar
from torch.utils.data import DataLoader
from utils.myfed import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true') 
    parser.add_argument('--gpu', type=str, default ='0')
    parser.add_argument('--num_workers', type=int, default = 8) 
    # data
    parser.add_argument('--dataset', type=str, default = 'cifar10')
    parser.add_argument('--datapath', type=str, default = '/data_local/xuangong/data/')
    # path
    parser.add_argument('--logfile', default='', type=str)
    parser.add_argument('--subpath', type=str, default ='') #subpath under localtraining folder to save central model

    # local training
    parser.add_argument('--N_parties', type=int, default = 20)
    parser.add_argument('--alpha', type=float, default = 1.0)
    parser.add_argument('--seed', type=int, default = 1)
    parser.add_argument('--C', type=float, default = 1) #percent of locals selected in each fed communication round
    parser.add_argument('--fedinitepochs', type=int, default = 20) #epochs of local training
    parser.add_argument('--batchsize', type=int, default =16)#128
    parser.add_argument('--lr', type=float, default = 0.0025)#0.025
    parser.add_argument('--lrmin', type=float, default = 0.001)
    parser.add_argument('--distill_droprate', type=float, default = 0)
    parser.add_argument('--optim', type=str, default ='ADAM')
    # fed setting
    parser.add_argument('--fedrounds', type=int, default = 200)
    parser.add_argument('--public_percent', type=float, default = 1.0) #ablation for c100 as public data
    parser.add_argument('--oneshot', action='store_true')
    parser.add_argument('--joint', action='store_true') #only valid when wo/ QN
    parser.add_argument('--quantify', type=float, default = 0.0) #when w/ QN
    parser.add_argument('--noisescale', type=float, default = 0.0) #when w/ QN

    # fed training param
    parser.add_argument('--disbatchsize', type=int, default = 512)  
    parser.add_argument('--localepochs', type=int, default = 10)
    parser.add_argument('--initepochs', type=int, default = 500)
    parser.add_argument('--initcentral', type=str, default = '')#ckpt used to init central model, import for co-distillation
    parser.add_argument('--wdecay', type=float, default = 0)
    parser.add_argument('--steps_round', type=int, default = 10000)
    parser.add_argument('--dis_lr', type=float, default = 1e-3) #1e-3
    parser.add_argument('--dis_lrmin', type=float, default = 1e-3) #1e-5
    parser.add_argument('--momentum', type=float, default = 0.9)
    
    #ensemble
    parser.add_argument('--voteout', action='store_true')
    parser.add_argument('--clscnt', type=int, default = 1) #local weight specific to class
    #loss
    parser.add_argument('--lossmode', type=str, default = 'l1') #kl or l1

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    
    handlers = [logging.StreamHandler()]
    if args.logfile:
        args.logfile = f'{datetime.now().strftime("%m%d%H%M")}'+args.logfile
    else:
        if args.joint:
            args.logfile = f'{datetime.now().strftime("%m%d%H%M")}_joint_a{args.alpha}s{args.seed}c{args.C}-sr{args.steps_round}'
        else:
            args.logfile = f'{datetime.now().strftime("%m%d%H%M")}_{args.dataset}_a{args.alpha}s{args.seed}c{args.C}_os{args.oneshot}_q{args.quantify}n{args.noisescale}'
    
    writer = SummaryWriter(comment=args.logfile)
    if not os.path.isdir('./logs'):
        os.mkdir('./logs')
    if args.debug:
        writer = None
        handlers.append(logging.FileHandler(
            f'./logs/debug.txt', mode='a'))
    else:
        handlers.append(logging.FileHandler(
            f'./logs/{args.logfile}.txt', mode='a'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers,     
    )
    logging.info(args)
    
    # 1. data
    args.datapath = os.path.expanduser(args.datapath)
    assert args.dataset=='cifar10' or args.dataset=='cifar100'
    publicdata = 'cifar100' if args.dataset=='cifar10' else 'imagenet'
    args.N_class = 10 if args.dataset=='cifar10' else 100
    priv_data, _, test_dataset, public_dataset, distill_loader = data_cifar.dirichlet_datasplit(
        args, privtype=args.dataset, publictype=publicdata, N_parties=20, online=not args.oneshot, public_percent=args.public_percent)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, sampler=None)
    
    ###########
    # 2. model
    logging.info("CREATE MODELS.......")
    gpu = [int(i) for i in range(torch.cuda.device_count())]
    logging.info(f'GPU: {args.gpu}')
    model = resnet8.ResNet8(num_classes=args.N_class).cuda()
    logging.info("totally {} paramerters".format(sum(x.numel() for x in model.parameters())))
    logging.info("Param size {}".format(np.sum(np.prod(x.size()) for name,x in model.named_parameters() if 'linear2' not in name)))
    if len(gpu)>1:
        model = nn.DataParallel(model, device_ids=gpu)
    
    # 3. fed training
    if args.quantify or args.noisescale:
        fed = FedKDwQN(model, distill_loader, priv_data, test_loader, writer, args)
        if args.oneshot:
            fed.update_distill_loader_wlocals(public_dataset)
            fed.distill_local_central_oneshot()
        else:
            fed.distill_local_central()
    else:
        fed = FedKD(model, distill_loader, priv_data, test_loader, writer, args)
        if args.joint:
            fed.distill_local_central_joint()
        elif args.oneshot:
            fed.update_distill_loader_wlocals(public_dataset)
            fed.distill_local_central_oneshot()
        else:
            fed.distill_local_central()
    
    if not args.debug:
        writer.close()


