import os
import numpy as np
import logging
import torch 
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import utils.utils as utils
import copy
import random
import loss.loss as Loss 
import dataset.data_cifar as data_cifar

class FedKD:
    def __init__(self, central, distil_loader, private_data, val_loader, 
                   writer, args, initpth=True, localmodel=None):
        # import ipdb; ipdb.set_trace()
        self.localmodel = localmodel
        self.central = central
        self.N_parties = args.N_parties
        self.distil_loader = distil_loader
        self.private_data = private_data
        self.val_loader = val_loader
        self.writer = writer
        self.args = args
        
        # ensemble and loss
        totalN = []
        for n in range(self.N_parties):
            totalN.append(private_data[n]['x'].shape[0])
        totalN = torch.tensor(totalN) #nlocal
        assert totalN.sum() == 50000
        self.totalN = totalN.cuda()#nlocal*1*1
        countN = np.zeros((self.N_parties, self.args.N_class))
        for n in range(self.N_parties):
            for m in range(self.args.N_class):
                countN[n][m] = (self.private_data[n]['y']==m).sum()
        assert countN.sum() == 50000
        self.countN = torch.tensor(countN).cuda()      
        self.locallist= list(range(0, self.N_parties))# local number list
        self.clscnt = args.clscnt# if ensemble is weighted by number of local sample
        self.voteout = args.voteout
        if args.lossmode == 'l1':
            self.criterion = torch.nn.L1Loss(reduce=True)
        elif args.lossmode=='kl': 
            self.criterion = Loss.kl_loss(T=3, singlelabel=True)
        else:
            raise ValueError
        
        # distill optimizer
        self.bestacc = 0
        self.best_statdict = self.central.state_dict()
        
        # import ipdb; ipdb.set_trace()
        #path to save ckpt
        self.rootdir = f'./checkpoints/{args.dataset}/a{self.args.alpha}+sd{self.args.seed}+e{self.args.initepochs}+b{self.args.batchsize}'
        if not os.path.isdir(self.rootdir):
            os.mkdir(self.rootdir)
        if initpth:
            if not args.subpath:
                if args.joint:
                    args.subpath = f'joint_c{args.C}_s{args.steps_round}_lr{args.dis_lr}_{args.dis_lrmin}'
                elif args.oneshot:
                    args.subpath = f'oneshot_c{args.C}_q{args.quantify}_n{args.noisescale}'
                else:
                    args.subpath = f'oneshot_c{args.C}_q{args.quantify}_n{args.noisescale}'
            self.savedir = os.path.join(self.rootdir, args.subpath)
            if not os.path.isdir(self.savedir):
                os.mkdir(self.savedir)
        self.init_locals(initpth, init_dir='')

    def init_locals(self, initpth=True, init_dir=''):
        epochs = self.args.initepochs
        if self.localmodel is None:
            self.localmodels = utils.copy_parties(self.N_parties, self.central)
        else:
            self.localmodels = utils.copy_parties(self.N_parties, self.localmodel)
        if not init_dir:
            init_dir = self.rootdir
        if initpth:
            for n in range(self.N_parties):
                savename = os.path.join(init_dir, str(n)+'.pt')
                if os.path.exists(savename):
                    #self.localmodels[n].load_state_dict(self.best_statdict, strict=True)
                    logging.info(f'Loading Local{n}......')
                    utils.load_dict(savename, self.localmodels[n])
                    acc = self.validate_model(self.localmodels[n])
                else:
                    logging.info(f'Init Local{n}, Epoch={epochs}......')
                    acc = self.trainLocal(savename, modelid=n)
                logging.info(f'Init Local{n}--Epoch={epochs}, Acc:{(acc):.2f}')
     
    def init_central(self):
        initname = os.path.join(self.rootdir, self.args.initcentral)
        if os.path.exists(initname):
            utils.load_dict(initname, self.central)
            acc = self.validate_model(self.central)
            logging.info(f'Init Central--Acc:{(acc):.2f}')
            self.bestacc = acc
            self.best_statdict = copy.deepcopy(self.central.state_dict())
        else:
            raise ValueError

    def distill_onemodel_batch(self, model, images, selectN, localweight, optimizer, usecentral=True):
        if usecentral:
            ensemble_logits = self.central(images).detach()
        else:
            #get local
            total_logits = []
            for n in selectN:
                tmodel = copy.deepcopy(self.localmodels[n])
                logits = tmodel(images).detach()
                total_logits.append(logits)
                del tmodel
            total_logits = torch.stack(total_logits) #nlocal*batch*ncls
            if self.voteout:
                ensemble_logits = Loss.weight_psedolabel(total_logits, self.countN[selectN], noweight=True).detach()
            else:    
                ensemble_logits = (total_logits*localweight).detach().sum(dim=0) #batch*ncls

        model.train()
        output = model(images)
        loss = self.criterion(output, ensemble_logits)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def distill_local_central(self):
        args = self.args
        if self.args.optim == 'SGD':
            optimizer = optim.SGD(
                self.central.parameters(), lr=self.args.dis_lr, momentum=args.momentum, weight_decay=args.wdecay)
        else:    
            optimizer = optim.Adam(
                self.central.parameters(), lr=self.args.dis_lr,  betas=(args.momentum, 0.999), weight_decay=args.wdecay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.fedrounds), eta_min=args.dis_lrmin,)
        
        savename = os.path.join(self.savedir, f'q{args.quantify}_n{args.noisescale}_{args.optim}_b{args.disbatchsize}_{args.dis_lr}_{args.fedrounds}_{args.dis_lrmin}_m{args.momentum}')
        bestacc = self.bestacc
        bestname = ''
        selectN = self.locallist
        if self.clscnt:
            countN = self.countN
            localweight = countN/countN.sum(dim=0)
            localweight = localweight.unsqueeze(dim=1)#nlocal*1*ncls
        else:
            localweight = 1.0*self.totalN/self.totalN.sum()
            localweight = localweight.unsqueeze(dim=1).unsqueeze(dim=2)#nlocal*1*1
        
        step = 0
        for epoch in range(0, args.fedrounds):
            for i, (images, _, _) in enumerate(self.distil_loader):
                images = images.cuda()
                countN = self.countN
                if self.args.C<1:
                    selectN = random.sample(self.locallist, int(args.C*self.N_parties))
                    #selectN = self.locallist[select]
                    countN = self.countN[selectN]
                    if self.clscnt:
                        localweight = countN/countN.sum(dim=0)#nlocal*nclass
                        localweight = localweight.unsqueeze(dim=1)#nlocal*1*ncls
                    else:
                        totalN = self.totalN[selectN]
                        localweight = 1.0*totalN/totalN.sum()
                        localweight = localweight.unsqueeze(dim=1).unsqueeze(dim=2)#nlocal*1*1
                
                loss = self.distill_onemodel_batch(self.central, images, selectN, localweight, optimizer, usecentral=False) 
                step += 1
                acc = self.validate_model(self.central)
                if self.writer is not None:
                    self.writer.add_scalar('loss', loss.item(), step)
                    self.writer.add_scalar('DisACC', acc, step)
                if acc>bestacc:
                    if bestname:
                        os.system(f'rm {bestname}')
                    bestacc = acc
                    bestname = f'{savename}_i{step}_{(bestacc):.2f}.pt'
                    torch.save(self.central.state_dict(), bestname)
                    logging.info(f'========Best...Iter{step},Epoch{epoch}, Acc{(acc):.2f}')
            scheduler.step()
            logging.info(f'Iter{step},Epoch{epoch}, Acc{(acc):.2f}')
      
    def distill_local_central_joint(self):
        if self.args.initcentral:
            self.init_central()
            usecentral = True
        else:
            usecentral = False
        if self.args.C==1:
            selectN = self.locallist
            countN = self.countN
            localweight = countN/countN.sum(dim=0)
            localweight = localweight.unsqueeze(dim=1)
        #optimizer
        self.totalSteps = int(self.args.steps_round*128/(self.args.disbatchsize))
        self.earlystopSteps = int(self.totalSteps/5)
        self.max_epochs_round = 1+int(self.totalSteps/len(self.distil_loader))

        self.localsteps = np.zeros(self.N_parties)
        local_optimizers = []
        local_schedulers = []
        for n in range(self.N_parties):
            if self.args.optim == 'SGD':
                optimizer = optim.SGD(
                    self.localmodels[n].parameters(), lr=self.args.dis_lr, momentum=self.args.momentum,  weight_decay=self.args.wdecay)
            else:    
                optimizer = optim.Adam(
                    self.localmodels[n].parameters(), lr=self.args.dis_lr,  betas=(self.args.momentum, 0.999), weight_decay=self.args.wdecay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(self.args.fedrounds), eta_min=self.args.dis_lrmin,)
            local_optimizers.append(optimizer)
            local_schedulers.append(scheduler)

        for roundd in range(1, 1+self.args.fedrounds):
            if self.args.C<1:    
                selectN = random.sample(self.locallist, int(self.args.C*self.N_parties))
                countN = self.countN[selectN]
                localweight = countN/countN.sum(dim=0)
                localweight = localweight.unsqueeze(dim=1)
            acc = self.validateLocalTeacher(selectN, localweight)
            logging.info(f'*****************Round{roundd},LocalAVG:{(acc):.2f}***********************')
            self.updateLocals(local_optimizers, roundd, selectN, localweight, usecentral=usecentral)
            # import ipdb; ipdb.set_trace()
            for n in selectN:
                local_schedulers[n].step()
            acc = self.validateLocalTeacher(selectN, localweight)
            logging.info(f'*****************Round{roundd},LocalAVG:{(acc):.2f}***********************')
            self.updateCentral(roundd, selectN, localweight)
            
    def updateLocals(self, optimizers, roundd, selectN, selectweight, usecentral=False): #only update for the selected
        for n in selectN:
            logging.info(f'---------------------Local-{n}------------------------')
            self.distill_onelocal(roundd, n, optimizers[n],  selectN, selectweight, usecentral=usecentral, writer=self.writer)
            
    def updateCentral(self, roundd, selectN, selectweight):
        #
        step = 0
        args = self.args
        earlyStop = utils.EarlyStop(self.earlystopSteps, self.totalSteps, bestacc=self.bestacc)
        self.best_statdict = copy.deepcopy(self.central.state_dict())
        
        if self.args.optim == 'SGD':
            optimizer = optim.SGD(
                self.central.parameters(), lr=self.args.dis_lr, momentum=self.args.momentum, weight_decay=args.wdecay)
        else:    
            optimizer = optim.Adam(
                self.central.parameters(), lr=self.args.dis_lr,  betas=(self.args.momentum, 0.99), weight_decay=args.wdecay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(self.totalSteps), eta_min=args.dis_lrmin,)

        for epoch in range(self.max_epochs_round):
            for i, (images, _, _) in enumerate(self.distil_loader):
                images = images.cuda()
                loss = self.distill_onemodel_batch(self.central, images, selectN, selectweight, optimizer, usecentral=False)
                acc = self.validate_model(self.central)
                step += 1
                stop, best = earlyStop.update(step, acc)
                if best:
                    logging.info(f'Iter{step}, best for now:{acc}')
                    self.best_statdict = copy.deepcopy(self.central.state_dict())
                if stop:
                    break
                else:
                    logging.info(f'===R{roundd}, Epoch:{epoch}/{self.max_epochs_round}, acc{acc}, best{earlyStop.bestacc}')
                    scheduler.step()
                    continue
            break
        #
        self.central.load_state_dict(self.best_statdict, strict=True)
        savename = os.path.join(self.savedir, f'r{roundd}_{(earlyStop.bestacc):.2f}.pt')
        torch.save(self.best_statdict, savename)
        logging.info(f'==================Round{roundd},Init{(self.bestacc):.2f}, Acc{(earlyStop.bestacc):.2f}====================')
        self.bestacc = earlyStop.bestacc
        if self.writer is not None:
            self.writer.add_scalar('DisACC', self.bestacc, roundd) 
    
    def distill_onelocal(self, roundd, modelid, optimizer, selectN, localweight, usecentral = False, writer=None):
        model = self.localmodels[modelid]
        initacc = self.validate_model(model)
        bestdict = copy.deepcopy(model.state_dict())
        earlyStop = utils.EarlyStop(self.earlystopSteps, self.totalSteps, bestacc=initacc)
        #
        savename = os.path.join(self.savedir, f'Local{modelid}_r{roundd}')
        writermark = f'Local{modelid}'
        step = 0
        globalstep = self.localsteps[modelid]
        bestname = ''
        for epoch in range(0, self.max_epochs_round):
            for i, (images, _, _) in enumerate(self.distil_loader):
                images = images.cuda()
                loss = self.distill_onemodel_batch(model, images, selectN, localweight, optimizer, usecentral=usecentral)
                step += 1
                globalstep += 1
                acc = self.validate_model(model)
                if writer is not None:
                    writer.add_scalar(writermark, acc, globalstep)
                stop, best = earlyStop.update(step, acc)
                if best:
                    if bestname:
                        os.system(f'rm {bestname}')
                    bestacc = acc
                    bestdict = copy.deepcopy(model.state_dict())
                    bestname = f'{savename}_i{int(globalstep):d}_{(bestacc):.2f}.pt'
                    torch.save(model.state_dict(), bestname)
                    logging.info(f'========Best...Iter{globalstep},Epoch{epoch}, Acc{(acc):.2f}')
                if stop:
                    break
            else:
                logging.info(f'Epoch:{epoch}/{self.max_epochs_round}, acc{acc}, best{earlyStop.bestacc}')
                #scheduler.step()
                continue
            break
        lastacc = self.validate_model(model)        
        model.load_state_dict(bestdict, strict=True)
        acc = self.validate_model(model)
        logging.info(f'R{roundd}, L{modelid}, ========Init{(initacc):.2f}====Final{(lastacc):.2f}====Best{(earlyStop.bestacc):.2f}')
        self.localsteps[modelid] = globalstep

    
    def validate_model(self, model):
        model.eval()
        testacc = utils.AverageMeter()
        with torch.no_grad():
            for i, (images, target, _) in enumerate(self.val_loader):
                images = images.cuda()
                target = target.cuda()
                output = model(images)
                acc, = utils.accuracy(output.detach(), target)
                testacc.update(acc)
        return testacc.avg    

    def validateLocalTeacher(self, selectN, localweight):   
        testacc = utils.AverageMeter()
        with torch.no_grad():
            for i, (images, target, _) in enumerate(self.val_loader):
                logits = []
                images = images.cuda()
                target = target.cuda()
                for n in selectN:
                    output = self.localmodels[n](images).detach()
                    logits.append(output)
                logits = torch.stack(logits)
                if self.voteout:
                    ensemble_logits = Loss.weight_psedolabel(logits, self.countN[selectN], noweight=True)
                else:
                    ensemble_logits = (logits*localweight).sum(dim=0)
                acc, = utils.accuracy(ensemble_logits, target)
                testacc.update(acc)
        return testacc.avg
    
    def trainLocal(self, savename, modelid=0):
        epochs = self.args.initepochs
        model = self.localmodels[modelid]
        tr_dataset = data_cifar.Dataset_fromarray(self.private_data[modelid]['x'],self.private_data[modelid]['y'], train=True, verbose=False)
        train_loader = DataLoader(
            dataset=tr_dataset, batch_size=self.args.batchsize, shuffle=True, num_workers=self.args.num_workers, sampler=None) 
        test_loader = self.val_loader
        args = self.args
        writer = self.writer
        datasize = len(tr_dataset)
        criterion = nn.CrossEntropyLoss() #include softmax
        optimizer = optim.SGD(model.parameters(), args.lr,
                                    momentum=0.9,
                                    weight_decay=3e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(epochs), eta_min=args.lrmin,)
        #
        bestacc = 0 
        bestname = ''
        for epoch in range(epochs):
            #train
            model.train()
            tracc = utils.AverageMeter()
            trloss = utils.AverageMeter()
            for i, (images, target, _) in enumerate(train_loader):
                images = images.cuda()
                target = target.cuda()
                output = model(images)
                # import ipdb; ipdb.set_trace()
                loss = criterion(output, target)
                acc,  = utils.accuracy(output, target)
                tracc.update(acc)
                trloss.update(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            logging.info(f'loss={trloss.avg}, acc={tracc.avg}')
            if writer is not None:
                writer.add_scalars(str(modelid)+'train', {'loss': trloss.avg}, epoch)
                writer.add_scalars(str(modelid)+'train', {'acc': tracc.avg}, epoch)
            #val
            model.eval()
            testacc = utils.AverageMeter()
            with torch.no_grad():
                for i, (images, target, _) in enumerate(test_loader):
                    images = images.cuda()
                    target = target.cuda()
                    output = model(images)
                    acc, = utils.accuracy(output, target)
                    testacc.update(acc)
                if writer is not None:
                    writer.add_scalar(str(modelid)+'testacc', testacc.avg, epoch)
                if testacc.avg > bestacc:
                    bestacc = testacc.avg
                    if bestname:
                        os.system(f'rm {bestname}')
                    bestname = f'{savename[:-3]}_{(bestacc):.2f}.pt'
                    torch.save(model.state_dict(), bestname)
                    os.system(f'cp {bestname} {savename}')
                logging.info(f'{modelid},Size={datasize},Epoch={epoch}: testacc={testacc.avg}, Best======{bestacc}======')
            #
            scheduler.step()

        return bestacc
       
    def update_distill_loader_wlocals(self, public_dataset):
        """
        save local prediction for one-shot distillation
        """
        total_logits = []
        for i, (images, _, idx) in enumerate(self.distil_loader):
            # import ipdb; ipdb.set_trace()
            images = images.cuda()
            batch_logits = []
            for n in self.locallist:
                tmodel = copy.deepcopy(self.localmodels[n])
                logits = tmodel(images).detach()
                batch_logits.append(logits)
                del tmodel
            batch_logits = torch.stack(batch_logits).cpu()#(nl, nb, ncls)
            total_logits.append(batch_logits)
        self.total_logits = torch.cat(total_logits,dim=1).permute(1,0,2) #(nsample, nl, ncls)
        if self.args.dataset=='cifar10':
            assert public_dataset.aug == False
            public_dataset.aug = True
        
        self.distil_loader = DataLoader(
            dataset=public_dataset, batch_size=self.args.disbatchsize, shuffle=True, 
            num_workers=self.args.num_workers, pin_memory=True, sampler=None)

    def distill_batch_oneshot(self, model, images, idx, selectN, localweight, optimizer):
        total_logits = self.total_logits[idx].permute(1,0,2) #nlocal*batch*ncls
        total_logits = total_logits[torch.tensor(selectN)].to(images.device) #nlocal*batch*ncls
        #if quantify

        if self.voteout:
            ensemble_logits, votemask = Loss.weight_psedolabel(total_logits, self.countN[selectN])
        else:    
            ensemble_logits = (total_logits*localweight).sum(dim=0) #batch*ncls
        #if noise

        model.train()
        central_logits = model(images)
        loss = self.criterion(central_logits, ensemble_logits)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

        
    def distill_local_central_oneshot(self):
        args = self.args
        if self.args.optim == 'SGD':
            optimizer = optim.SGD(
                self.central.parameters(), lr=self.args.dis_lr, momentum=args.momentum, weight_decay=args.wdecay)
        else:    
            optimizer = optim.Adam(
                self.central.parameters(), lr=self.args.dis_lr,  betas=(args.momentum, 0.999), weight_decay=args.wdecay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.fedrounds), eta_min=args.dis_lrmin,)
        
        savename = os.path.join(self.savedir, f'osp{args.public_percent}_q{args.quantify}_n{args.noisescale}_{args.optim}_b{args.disbatchsize}_{args.dis_lr}_{args.fedrounds}_{args.dis_lrmin}_m{args.momentum}')
        bestacc = self.bestacc
        bestname = ''
        selectN = self.locallist
        if self.clscnt:
            countN = self.countN
            localweight = countN/countN.sum(dim=0)
            localweight = localweight.unsqueeze(dim=1)#nlocal*1*ncls
        else:
            localweight = 1.0*self.totalN/self.totalN.sum()
            localweight = localweight.unsqueeze(dim=1).unsqueeze(dim=2)#nlocal*1*1
        
        step = 0
        for epoch in range(0, args.fedrounds):
            for i, (images, _, idx) in enumerate(self.distil_loader):
                images = images.cuda()
                countN = self.countN
                if self.args.C<1:
                    selectN = random.sample(self.locallist, int(args.C*self.N_parties))
                    countN = self.countN[selectN]
                    if self.clscnt:
                        localweight = countN/countN.sum(dim=0)#nlocal*nclass
                        localweight = localweight.unsqueeze(dim=1)#nlocal*1*ncls
                    else:
                        totalN = self.totalN[selectN]
                        localweight = 1.0*totalN/totalN.sum()
                        localweight = localweight.unsqueeze(dim=1).unsqueeze(dim=2)#nlocal*1*1
                
                loss = self.distill_batch_oneshot(self.central, images, idx, selectN, localweight, optimizer)
                step += 1
                acc = self.validate_model(self.central)
                if self.writer is not None:
                    self.writer.add_scalar('loss', loss.item(), step)
                    self.writer.add_scalar('DisACC', acc, step)
                if acc>bestacc:
                    if bestname:
                        os.system(f'rm {bestname}')
                    bestacc = acc
                    bestname = f'{savename}_i{step}_{(bestacc):.2f}.pt'
                    torch.save(self.central.state_dict(), bestname)
                    logging.info(f'========Best...Iter{step},Epoch{epoch}, Acc{(acc):.2f}')
            scheduler.step()
            logging.info(f'Iter{step},Epoch{epoch}, Acc{(acc):.2f}')
        

class FedKDwQN(FedKD):
    def distill_onemodel_batch(self, model, images, selectN, localweight, optimizer, usecentral=True):
        if self.args.noisescale:
            laplace = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([self.args.noisescale]))
        if usecentral:
            ensemble_logits = self.central(images).detach()
        else:
            #get local
            total_logits = []
            for n in selectN:
                tmodel = copy.deepcopy(self.localmodels[n])
                logits = tmodel(images).detach()
                total_logits.append(logits)
                del tmodel
            total_logits = torch.stack(total_logits) #nlocal*batch*ncls
            if self.args.quantify:
                #quantify orginal =100
                logits_max = total_logits.abs().max()
                norm_total_logits = total_logits/logits_max #(-1,1)
                norm_total_logits_q200 = (norm_total_logits*self.args.quantify+0.5).int()#(-99,100)
                #de-quantify
                re_total_logits = (norm_total_logits_q200/self.args.quantify)*logits_max
                # import ipdb; ipdb.set_trace()
                total_logits = re_total_logits
            #
            if self.voteout:
                ensemble_logits = Loss.weight_psedolabel(total_logits, self.countN[selectN], noweight=True).detach()
            else:    
                ensemble_logits = (total_logits*localweight).detach().sum(dim=0) #batch*ncls
            if self.args.noisescale:
                nb, nc = ensemble_logits.shape
                noise = laplace.sample((nb,nc)).to(ensemble_logits.device).squeeze(dim=-1)
                # import ipdb; ipdb.set_trace()
                ensemble_logits += noise
        model.train()
        output = model(images)
        loss = self.criterion(output, ensemble_logits)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def distill_batch_oneshot(self, model, images, idx, selectN, localweight, optimizer):
        if self.args.noisescale:
            laplace = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([self.args.noisescale]))
        
        total_logits = self.total_logits[idx].permute(1,0,2) #nlocal*batch*ncls
        total_logits = total_logits[torch.tensor(selectN)].to(images.device) #nlocal*batch*ncls
        if self.args.quantify:
            #quantify orginal =100
            logits_max = total_logits.abs().max()
            norm_total_logits = total_logits/logits_max #(-1,1)
            norm_total_logits_q200 = (norm_total_logits*self.args.quantify+0.5).int()#(-99,100)
            #de-quantify
            re_total_logits = (norm_total_logits_q200/self.args.quantify)*logits_max
            # import ipdb; ipdb.set_trace()
            total_logits = re_total_logits

        if self.voteout:
            ensemble_logits, votemask = Loss.weight_psedolabel(total_logits, self.countN[selectN])
        else:    
            ensemble_logits = (total_logits*localweight).sum(dim=0) #batch*ncls
        if self.args.noisescale:
            nb, nc = ensemble_logits.shape
            # noise = torch.normal(mean=0.0, std=self.args.noisestd, size=(nb, nc)).to(logits.device)
            noise = laplace.sample((nb,nc)).to(ensemble_logits.device).squeeze(dim=-1)
            # import ipdb; ipdb.set_trace()
            ensemble_logits += noise

        model.train()
        central_logits = model(images)
        loss = self.criterion(central_logits, ensemble_logits)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss