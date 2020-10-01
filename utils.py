from functools import partial
from typing import Iterable
from torch import nn
import torch
from torch import tensor
import math

def compose(x,funcs,order_key = '_order',**kwargs):
    if funcs is None: return x
    funcs = listify(funcs)
    for f in sorted(funcs, key = lambda i: getattr(i,order_key,0)): x = f(x, **kwargs)
    return x


def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]



class ListContainer():
    def __init__(self,items): self.items = listify(items)

    def __getitem__(self,idx):
        if isinstance(idx, (int,slice)): return self.items[idx]
        elif isinstance(idx[0], bool):
            assert len(idx) == len(self)
            return [o for m,o in zip(idx,self.items) if m]
        return [self.items[i] for i in idx]

    def __len__(self): return len(self.items)

    def __iter__(self): return iter(self.items)

    def __setitem__(self,i,o): self.items[i] = o
    def __delitem__(self,i): del(self.items[i])

    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self)>10: res = res[:-1]+ '...]'
        return res


def accuracy(out,targ): return (torch.argmax(out,dim = 1) == targ).float().mean()

class FlatLoss(nn.Module):
    def __init__(self, func, reduction=None):
        super().__init__()
        self.func = func
        self.reduction = reduction if reduction is not None else self.func.reduction
        setattr(self.func, 'reduction', self.reduction)
        
    def __call__(self,inp,targ):
        return self.func(inp.contiguous().view(-1,inp.size(-1)), targ.contiguous().view(-1))



class AvgStats():
    def __init__(self, metrics, in_train): self.metrics,self.in_train = listify(metrics),in_train

    def reset(self):
        self.tot_loss,self.count = 0.,0
        self.tot_mets = [0.] * len(self.metrics)

    @property
    def all_stats(self): return [self.tot_loss.item()] + self.tot_mets
    @property
    def avg_stats(self): return [o/self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count: return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i,m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn

class AvgStatsCap():
    def __init__(self, metrics, in_train): self.metrics,self.in_train = listify(metrics),in_train

    def reset(self):
        self.tot_loss,self.count = 0.,0
        self.tot_mets = [0.] * len(self.metrics)

    @property
    def all_stats(self): return [self.tot_loss.item()] + self.tot_mets
    @property
    def avg_stats(self): return [o/self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count: return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, run):
        bn = run.xb[0].shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i,m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn

def cap_accuracy(out,targ): 
    a = 0
    for i in range(out.size(0)):
        a += accuracy(out[i, :,:], targ[i, :])
    return a/out.size(0)

def extract_mean_std(x):
    return (x.mean(), x.std())

def normalise(x,mean,std):
    x = (x - mean[...,None,None])/std[...,None,None]
    return x

def cap_normalise(x,mean,std):
    im = x[0]
    im = (im - mean[...,None,None])/std[...,None,None]
    x = (im,x[1])
    return x


def annealer(f):
    def _inner(start,end): return partial(f,start,end)
    return _inner

@annealer
def lin_scheduler(start,end,pos): return start + pos*(end-start)

@annealer
def cos_scheduler(start,end,pos): return ((end-start)/2)*math.cos((pos-1)*math.pi) + ((end+start)/2)

@annealer
def exp_scheduler(start,end,pos): return start*(end/start)**pos

def combine_scheds(pcts, scheds):
    assert int(sum(pcts)) == 1
    assert len(pcts) == len(scheds)
    pcts = tensor([0] + pcts)
    pcts = torch.cumsum(pcts,0)
    def _inner(pos):
        idx = (pos >= pcts).sum().int()
        idx -= 2 if idx == len(pcts) else 1
        npos = (pos - pcts[idx])/(pcts[idx+1] - pcts[idx])
        return scheds[idx](npos)
    return _inner


def create_phases(*fracs):
    fracs = listify(fracs)
    if sum(fracs) == 1: return fracs
    return fracs + [1-sum(fracs)]

def cos_annealer(lr1,lr2,lr3):
    return [cos_scheduler(lr1,lr2), cos_scheduler(lr2,lr3)]
