from utils import ListContainer, compose,partial
import mimetypes
import os
import torch
import PIL
from pathlib import Path
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
import pandas as pd


image_ext = [k for k,v in mimetypes.types_map.items() if 'image' in v]

image_ext = [i.strip('.') for i in image_ext]

def get_files(path, ext = image_ext, recurse = False,include = None):
    if recurse:
        res = []
        for i, (p,d,f) in enumerate(os.walk(path)):
            if i == 0 and include is not None:
                d[:] = include
                continue
            if f:
                if f[0].split('.')[-1].lower() in ext: res += get_files(Path(p),ext = ext,
                                                    recurse = False,include = None)
        return res
    else:
        return [path/i.name for i in os.scandir(path) if i.name.split('.')[-1].lower() in ext]


class ItemList(ListContainer):
    def __init__(self,path = None,items=None,how_many = None,tfms = None):
        if how_many is None: how_many = len(items)
        super().__init__(items[:how_many])
        self.path = path
        self.tfs = tfms

    def __repr__(self): return f'{super().__repr__()}'

    def new(self, items, tfms = None,cls=None):
        if cls is None: cls=self.__class__
        if not tfms: tfms = self.tfs
        return cls(self.path,items,tfms=tfms)

    def get(self,o): return o

    def comp(self,o): return compose(self.get(o), self.tfs)

    def __getitem__(self,i):
        res = super().__getitem__(i)
        if isinstance(res,list): return [self.comp(o) for o in res]
        return self.comp(res)



class ImageList(ItemList):
    @classmethod
    def from_path(cls,path,ext = None, recurse = True,include = None,**kwargs):
        if ext is None: ext = image_ext
        return cls(path, get_files(path,ext = ext,recurse = recurse, include = include), **kwargs)

    def get(self,o): return PIL.Image.open(o)


def split_from_func(items,f):
    mask = [f(o) for o in items]
    t = [i for i,m in zip(items,mask) if not m]
    v = [i for i,m in zip(items,mask) if m]
    return t,v

class SplitDataset():
    def __init__(self,train,valid):
        self.train,self.valid = train,valid

    @classmethod
    def from_func(cls,itemlist,func):
        vtfs = cls.give_valid_tfs(itemlist.tfs)
        t,v = split_from_func(itemlist.items,func)
        t,v = itemlist.new(t), itemlist.new(v,tfms = vtfs)
        return cls(t,v)

    @classmethod
    def from_ratio(cls,itemlist,ratio):
        l = np.random.permutation(itemlist.items)
        vtfms = cls.give_valid_tfs(itemlist.tfs)
        v,t = l[:int(len(l)*ratio)],l[int(len(l)*ratio):]
        t,v = itemlist.new(t), itemlist.new(v,tfms = vtfms)
        return cls(t,v)

    @staticmethod
    def give_valid_tfs(tfms): return [i for i in tfms if getattr(i, '_valid',True)]

    def __repr__(self): return f'{self.__class__.__name__}\nTrain: {self.train}\nValid: {self.valid}\n'



def make_unique(l):
    l = list(OrderedDict.fromkeys(l).keys())
    return l

def label_func(df,column_name,o):
    o = o.name
    return column_name if df.loc[o][column_name] ==1 else 'Not ' + column_name

def label_all(il,f,cls = ItemList): return cls(items = [f(o) for o in il.items])

class Processor():
    def __init__(self):
        self.vocab = None
#         self.word_label = None

    def __call__(self,items):
        if self.vocab is None:
            self.vocab = make_unique(items)
            self.otoi = {v:k for k,v in enumerate(self.vocab)}
        return [self.proc1(i) for i in items]

    def proc1(self,i): return self.otoi[i]
    def deproc1(self,idx): return self.vocab[idx]
    def deprocess(self,idxs): return [self.deproc1(i) for i in idxs]



class LabeledList():
    def process(self,il,processor): return il.new(compose(il.items,processor))

    def __init__(self,x,y,proc_x = None,proc_y = None):
        self.x,self.y = self.process(x,proc_x),self.process(y,proc_y)
        self.px,self.py = proc_x, proc_y

    def __getitem__(self,i):return self.x[i],self.y[i]

    def __repr__(self): return f'{self.__class__.__name__}\nx: {self.x}\ny: {self.y}\n'

    def __len__(self): return len(self.x)

    def x_obj(self,i): return self.decode(self.x[i],self.px)
    def y_obj(self,i): return self.decode(self.y[i],self.py)

    def decode(self,i,p):
        if isinstance(i,int): return p.deproc1(i)
        else: return p.deprocess(i)

    @classmethod
    def label_by_df(cls,items,csv_path,index_name,column_name,proc_x = None,proc_y = None):
        df = pd.read_csv(csv_path)
        df.set_index(index_name,inplace = True)
        return cls(items,label_all(items,partial(label_func,df,column_name)), proc_x = proc_x, proc_y = proc_y)

    @classmethod
    def label_by_func(cls,items,func,proc_x = None,proc_y = None):
        return cls(items, label_all(items,func), proc_x = proc_x, proc_y = proc_y)


def label_train_valid(sd,func = None,csv_path = None,index_name = None,column_name = None,procx = None,procy = None):
    if csv_path:
        train = LabeledList.label_by_df(sd.train, csv_path=csv_path, index_name=index_name,column_name=column_name, proc_x=procx,proc_y=procy)
        valid = LabeledList.label_by_df(sd.valid, csv_path=csv_path, index_name=index_name,column_name=column_name, proc_x=procx,proc_y=procy)
    else:
        train = LabeledList.label_by_func(sd.train, func = func, proc_x=procx,proc_y=procy)
        valid = LabeledList.label_by_func(sd.valid, func = func, proc_x=procx,proc_y=procy)
    return SplitDataset(train,valid)

class DataBunch():
    def __init__(self,train_dl,valid_dl):
        self.train_dl,self.valid_dl = train_dl,valid_dl

    @property
    def train_ds(self): return self.train_dl.dataset

    @property
    def valid_ds(self): return self.valid_dl.dataset

    @property
    def path(self): return self.train_ds.x.path



def get_dls(train_ds,valid_ds,train_bs,valid_bs,shuffle = True,nw = 0):
    if valid_bs is None:
        valid_bs = train_bs
    return DataLoader(train_ds,batch_size=  train_bs,shuffle = shuffle,num_workers = nw), DataLoader(valid_ds,batch_size=valid_bs,num_workers=nw)

def get_cap_dls(labelled_list, train_bs, valid_bs,collate_fn = None, shuffle = True, num_workers = 0,flip = False):
    if valid_bs is None:
        valid_bs = train_bs
    if collate_fn is not None: 
        collate_fn = partial(collate_fn, pad_idx = labelled_list.train.py.tokenizer.pad_token_id, flip = flip)
    return (DataLoader(labelled_list.train, batch_size= train_bs, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers),
              DataLoader(labelled_list.valid, batch_size= valid_bs, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers))


def databunchify(sd,train_bs,valid_bs = None,**kwargs):
    return DataBunch(*get_dls(sd.train, sd.valid,train_bs,valid_bs,**kwargs))


SplitDataset.to_databunch = databunchify


#export
class Tokprocessor():
    def __init__(self,tokenizer,max_len = 49):
        self.tokenizer = tokenizer
        if self.tokenizer.eos_token is None or self.tokenizer.pad_token is None:
            self._add_special_tokens()
        
        self.bos_tok = self.tokenizer.bos_token
        self.eos_tok = self.tokenizer.eos_token
        self.ml = max_len
        
    def __call__(self, labels):
        for i in range(len(labels)):
            labels[i] = torch.tensor(self.tokenizer.encode(self.bos_tok + " " + labels[i] + " " + self.eos_tok, 
                                              max_length = self.ml, padding = 'max_length', truncation = True), 
                                     dtype = torch.int32)
        return torch.stack(labels, dim = 0)
    
    def decode(self,tokens):
        return self.tokenizer.decode(tokens,skip_special_tokens = True)
    
    def _add_special_tokens(self):
        self.tokenizer.add_special_tokens({'bos_token':'<BOS>', 'eos_token':'<EOS>', 'pad_token':'<PAD>'})

def decode_tokens(ll, i,processor):
    if isinstance(i, list): return processor.decode(i)
    else: 
        a = []
        for k in i: a.append(processor.decode(k))
    return a

# LabeledList.decode = decode_tokens