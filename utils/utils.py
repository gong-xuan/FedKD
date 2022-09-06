import torch, os, copy
import numpy as np
from sklearn import metrics
import logging

def copy_parties(n_local, model):
    parties = []
    for n in range(n_local):
        model = clone_model(model)
        parties.append(model)
    return parties


# def remove_lastlayer(model):
#     list(model.modules())
#     my_model = (nn.Sequential)(*list(model.modules())[:-1])
#     state_dict = {k:v for k, v in model.state_dict().items() if k in my_model.state_dict()}
#     my_model.load_state_dict(state_dict)
#     return my_model


def clone_model(model):
    my_model = copy.deepcopy(model)
    return my_model

def load_dict(savepath, model):
    pth = torch.load(savepath)
    is_data_parallel = isinstance(model, torch.nn.DataParallel)
    new_pth = {}
    for k, v in pth.items():
        if 'module' in k:
            if is_data_parallel: # saved multi-gpu, current multi-gpu
                new_pth[k] = v
            else: # saved multi-gpu, current 1-gpu 
                new_pth[k.replace('module.', '')] = v
        else: 
            if is_data_parallel: # saved 1-gpu, current multi-gpu
                new_pth['module.'+k] = v
            else: # saved 1-gpu, current 1-gpu
                new_pth[k] = v 
    m, u = model.load_state_dict(new_pth, strict=False)
    if m:
        logging.info('Missing: '+' '.join(m))
    if u:
        logging.info('Unexpected: '+' '.join(u))
    return

class EarlyStop(object):
    def __init__(self, max_plateau, totaliters, bestacc=0, min_increase=0.1):
        self.bestacc = bestacc
        self.bestiter = 0
        self.totaliters = totaliters
        self.max_plateau = max_plateau
        self.min_increase = min_increase

    def update(self, itern, acc):
        best = (acc-self.bestacc>self.min_increase)
        if best:
            self.bestacc = acc
            self.bestiter = itern
        if itern==self.totaliters:
            logging.info(f'Total: {itern}...Bestiter{self.bestiter}')
            return True, best
        elif itern-self.bestiter == self.max_plateau:
            logging.info(f'Plateau: {itern}-{self.bestiter}...Total{self.totaliters}')
            return True, best
        else:
            return False, best


class AverageMeter(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    usage:
    prec1,prec5=accuracy(output,target,topk=(1,5))
    """
    maxk = max(topk)
    batchsize = target.size(0)
    if len(target.shape) == 2: #multil label
        output_mask = output > 0.5
        correct = (output_mask == target).sum()
        return [100.0*correct.float() / target.numel()]
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batchsize).item())
    return res

def get_model_size(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


def save_checkpoint(model, filename='model_best.pth.tar'):
    torch.save(model.state_dict(), filename)


def load_local_statedicts(model, filename):
    if os.path.isfile(filename):
        print("=>loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_local_statedicts(checkpoint)
        #model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=>no checkpoint found at '{}'".format(filename))

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def auc_multiclass(prob, label):
    auc = []
    for n in range(label.shape[1]):
        pred = prob[:,n]
        true = label[:,n]
        fpr, tpr, thresh = metrics.roc_curve(true, pred, pos_label=1)
        auc.append(metrics.auc(fpr, tpr))
    return auc


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if __name__ == '__main__':
    x = torch.rand(32,3,16,16)
    y = torch.zeros(32,5)
    output = torch.rand(32,5)
    mixed_x, y_a, y_b, lam = mixup_data(x,y, use_cuda=False)
    loss_function = mixup_criterion(y_a, y_b, lam)
    criterion = torch.nn.BCELoss()
    loss = loss_function(criterion, output)
    import ipdb; ipdb.set_trace()