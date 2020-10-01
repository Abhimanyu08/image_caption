from torch import nn
from torch import tensor
import torch,math
import torch.nn.functional as F
from torch.cuda import amp
from functools import partial

# class LayerNorm(nn.Module):
#     def __init__(self, nc,eps=1e-06):
#         super().__init__()
#         self.mult = nn.Parameter(torch.ones(1,1,nc))
#         self.add = nn.Parameter(torch.zeros(1,1,nc))
#         self.eps = eps

#     def forward(self,x):
#         m,s = x.mean(dim=-1, keepdim = True), x.std(dim=-1, keepdim = True)
#         return (self.mult*(x-m)/(s+self.eps))+self.add

class Generator(nn.Module):
    def __init__(self,nh, vocab_size):
        super().__init__()
        self.lin = nn.Linear(nh, vocab_size)

    def forward(self,x):
        return F.log_softmax(self.lin(x), dim = -1)

def calculate_attention(Q,K,V, mask = None, dropout = None):
    dk = V.size(-1)
    wgts = Q@K.transpose(-2,-1)/math.sqrt(dk)
    if mask is not None:
        mask = mask.unsqueeze(1).repeat(1, V.size(1), 1,1)
        wgts.masked_fill_(mask, -1e9)
    wgts = F.softmax(wgts, dim =-1)
    if dropout is not None:
        wgts = dropout(wgts)
    return wgts@V

class multi_headed_attention(nn.Module):
    def __init__(self,d_model=512,h=8, dropout=0.1):
        super().__init__()
        self.h = h
        self.dk = d_model//self.h
        self.lins = nn.ModuleList([nn.Linear(d_model,d_model)]*4)
        self.drop = nn.Dropout(p = dropout)

    def forward(self,q,k,v,mask=None):
        nb = q.size(0)

        q,k,v = [lin(x).view(nb,-1,self.h,self.dk).transpose(1,2) for lin,x in zip(self.lins, (q,k,v))]
        
        x = calculate_attention(q,k,v,mask = mask,dropout= self.drop)
        x = x.transpose(1,2).contiguous().view(nb, -1, self.h*self.dk)
        return self.lins[-1](x)

class FeedForward(nn.Module):
    def __init__(self,d_model = 512,d_ff=2048,dropout = 0.1, activation = 'gelu'):
        super().__init__()
        self.ffnn1 = nn.Linear(d_model,d_ff)
        self.ffnn2 = nn.Linear(d_ff,d_model)
        self.drop = nn.Dropout(p = dropout)
        self.ac = getattr(F, activation)
    def forward(self,x):
        return self.ffnn2(self.drop(self.ac(self.ffnn1(x))))

class self_attention_plus_feedforward(nn.Module):
    def __init__(self,decoder = False,d_model = 512, h=8,d_ff=2048,ff_drop = 0.1,attn_drop  = 0.1, activation = 'gelu'):
        super().__init__()
        self.attention = multi_headed_attention(d_model,h,attn_drop)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=ff_drop, activation= activation)
        self.lns = nn.ModuleList([nn.LayerNorm(d_model)]*2) if not decoder else nn.ModuleList([nn.LayerNorm(d_model)]*3)
        self.dec = decoder

    def forward(self,x,*args,mask_in=None,mask_out = None):
        x = self.lns[0](x + self.attention(x,x,x, mask = mask_in))
        if self.dec:
            ar = (x,)+args
            x = self.lns[1](x+self.attention(*ar, mask= mask_out))
        x = self.lns[-1](x + self.feed_forward(x))
        return x

class Xcoder(nn.Module):
    def __init__(self, dec = False, N = 6, d_model = 512, h = 8, d_ff = 2048, ff_drop = 0.1, attn_drop = 0.1,
    activation = 'gelu'):
        super().__init__()
        layer = self_attention_plus_feedforward(dec, d_model,h,d_ff,ff_drop,attn_drop, activation = activation)
        self.layers = nn.ModuleList([layer]*N)

    def forward(self,x,*args, mask_in = None,mask_out = None):
        for i in self.layers:
            x = i(x, *args, mask_in = mask_in,mask_out = mask_out)
        return x

def attention_mask(x):
    sz = (x.size(1),)*2
    mat = (torch.triu(torch.ones(*sz), diagonal=1)).bool()
    mat = mat.to(x.device)
    return mat[None]

def get_padding_mask(inp, pad_idx:int=1):
    bs,s = inp.size()
    mat = (inp == pad_idx).unsqueeze(1)
    mat = mat.expand(bs,s,s)
    return mat.to(inp.device)

class WordEmbedding(nn.Module):
    def __init__(self,vocab_size, d_model, padding_index):
        super().__init__()
        self.vs = vocab_size
        self.dm = d_model
        self.emb = nn.Embedding(self.vs,self.dm,padding_idx= padding_index)


    def forward(self,x):
        return self.emb(x)*math.sqrt(self.dm)


class PositionalEncoding(nn.Module):
    "Encode the position with a sinusoid."
    def __init__(self, d_model=512, max_len = 5000,dropout = 0):
        super().__init__()
        self.drop = nn.Dropout(p = dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return self.drop(x)

class Transformer(nn.Module):
    def __init__(self,N=6,word_emb = WordEmbedding, pos_emb = PositionalEncoding, gen = Generator,
                 input_vocab_size = 50000, targ_vocab_size = 50000, d_model = 512, h = 8, d_ff = 2048,
                 max_len = 5000, encoding_drop = 0.1, attn_drop = 0.1, ff_drop = 0.1, activation = 'gelu'):
        super().__init__()
        self.enc = Xcoder(False, N,d_model, h, d_ff,ff_drop, attn_drop,activation=activation)
        self.dec = Xcoder(True, N,d_model, h, d_ff,ff_drop, attn_drop, activation=activation)
        self.inp_emb = word_emb(input_vocab_size,d_model)
        self.pos_enc = pos_emb(d_model,max_len, encoding_drop)
        self.targ_emb = word_emb(targ_vocab_size, d_model)
        self.gen = gen(d_model, targ_vocab_size)

        self._init_parameters()

    def forward(self,x,o):
        pad_inp_mask = get_padding_mask(x)
        pad_attn_mask = get_padding_mask(o) + attention_mask(o)
        x = self.pos_enc(self.inp_emb(x))
        x = self.enc(x,mask_in = pad_inp_mask)
        memory = x
        o = self.pos_enc(self.targ_emb(o))
        o = self.dec(o,memory,memory,mask_in = pad_attn_mask,mask_out = pad_inp_mask)
        o = self.gen(o)
        return o

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
