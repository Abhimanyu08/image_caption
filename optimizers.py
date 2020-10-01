import torch
from utils import compose,listify,partial





class Optimizer():
    def __init__(self,params,steppers,**defaults):
        self.params = list(params)
        if not isinstance(self.params[0], list): self.params = [self.params]
        self.steppers = listify(steppers)
        update(defaults, self.steppers, get_defaults)
        self.hypers = [{**defaults} for p in self.params]


    def ret_params(self):
        return [(p,h) for pg,h in zip(self.params, self.hypers) for p in pg if p.grad is not None]

    def zero_grad(self):
        for p,_ in self.ret_params():
            p.grad.detach_()
            p.grad.zero_()

    def step(self):
        for p,h in self.ret_params(): compose(p, self.steppers, **h)



def get_defaults(o): return getattr(o, '_defaults', {})

def update(dic,obj,f):
    for o in obj:
#         k = getattr(o,p,{})
        for i,j in f(o).items():
            if i not in dic: dic[i] = j



class StatefulOptimizer(Optimizer):
    def __init__(self,params,steppers,stats = None,**defaults):
        self.stats = listify(stats)
        update(defaults, self.stats, get_defaults)
        super().__init__(params,steppers,**defaults)
        self.state = {}

    def step(self):
        for p,h in self.ret_params():
            if p not in self.state:
                self.state[p] = {}
                update(self.state[p], self.stats, lambda o: o.init_state(p))
            state = self.state[p]
            for stat in self.stats: state = stat.update(state,p,**h)
            compose(p,self.steppers,**state, **h)
            self.state[p] = state


def sgd_step(p,lr,**kw):
    p.data.sub_(lr,p.grad.data)
    return p


def weight_decay(p,lr,wd,**kwargs):
    bias = getattr(p,'bias', None)
    if not bias:
        p.data.mul_(1 - wd*lr)
    return p

weight_decay._defaults = dict(wd = 0.)

class Stat():
    _defaults = {}
    def init_state(self,p): raise NotImplementedError
    def update(self,state,p,**kw): raise NotImplementedError


class Step(Stat):
    def init_state(self,p): return {'step': 0}
    def update(self,state,p,**kw): 
        state['step'] += 1
        return state


class LookAheadStat(Stat):
    def init_state(self,p): return {'slow_wgts': p.data.clone().detach()}
    def update(self, state, p, k,**kwargs):
        if state['step'] %k == 1 and state['step'] > k:
            state['slow_wgts'] = p.data.clone().detach()
        return state

def lookahead_step(p, alpha, k, slow_wgts: torch.Tensor, step,**kwargs):
    if step % k == 0:
        slow_wgts.data.add_(p.data - slow_wgts, alpha = alpha)
        p.data.copy_(slow_wgts)
    return p

lookahead_step._order = 10
lookahead_step._defaults = {'alpha':0.8, 'k': 10}

class AverageGrad(Stat):
    _defaults = dict(mom = 0.9)
    def __init__(self,dampening = True): self.damp = dampening
    def init_state(self,p): return {'grad_avg': torch.zeros_like(p.grad.data)}
    def update(self,state,p,mom,**kwargs):
        state['mom_damp'] = 1.0-mom if self.damp else 1.
        state['grad_avg'].mul_(mom).add_(p.grad.data,alpha = state['mom_damp'])
        return state

def momentum_step(p,grad_avg,lr,**kwargs):
    p.data.sub_(grad_avg, alpha = lr)
    return p

class AvgSquareGrad(Stat):
    _defaults = dict(sq_mom = 0.99)
    def __init__(self,dampening = False): self.damp = dampening
    def init_state(self,p): return {'grad_sqr': torch.zeros_like(p.grad.data)}
    def update(self,state,p,sq_mom,**kw):
        state['sqr_damp'] = 1-sq_mom if self.damp else 1.
        state['grad_sqr'].mul_(sq_mom).addcmul_(p.grad.data, p.grad.data, value = state['sqr_damp'])
        return state

def debias(mom,damp,step): 
        return  damp * (1 - mom**step) / (1-mom)

def adam_step(p,lr,grad_avg,grad_sqr,mom,sq_mom,mom_damp,sqr_damp,step,eps,**kw):
    _defaults = dict(eps = 1e-10)
    d1 = debias(mom,mom_damp,step)
    d2 = debias(sq_mom,sqr_damp,step)
    p.data.addcdiv_(-lr/d1, grad_avg, (grad_sqr/d2).sqrt() + eps)
    return p

def lamb_step(p,lr,mom,sq_mom,mom_damp,sqr_damp,grad_avg, grad_sqr,wd,step,eps,**kw):
    d1 = debias(mom, mom_damp, step)
    d2 = debias(sq_mom, sqr_damp, step)
    r1 = p.data.pow(2).mean().sqrt()
    s = (grad_avg/d1)/((grad_sqr/d2) + eps).sqrt() + wd*p.data
    r2 = s.pow(2).mean().sqrt()
    p.data.sub_(lr*min(r1/r2,10),s)
    return p

lamb_step._defaults = dict(eps = 1e-6, wd = 0.)

def l2_norm(p): return p.pow(2).mean().sqrt()

class lars_stat(Stat):
    _defaults = dict(mom = 0.9, wd = 1e-4)
    def init_state(self,p): return {'lar_avg': torch.zeros_like(p.grad.data)}
    def update(self,state,p,lr,mom,wd,**kw):
        lamb = l2_norm(p.data)/(l2_norm(p.grad.data) + wd*l2_norm(p.data))
        state['lar_avg'].mul_(mom).add_(lr*lamb*(p.grad.data + wd*p.data))
        return state

def lars_step(p,lar_avg,**kw):
    p.data.sub_(lar_avg)
    return p
