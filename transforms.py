import random
from PIL import ImageEnhance
import numpy as np
import PIL
import torch
from utils import listify
import math
class Transform(): _order,_valid = 0, True

def into_rgb(x): return x.convert('RGB')

class ResizeFixed(Transform):
    _order=10
    def __init__(self,size):
        if isinstance(size,int): size=(size,size)
        self.size = size

    def __call__(self, item): return item.resize(self.size, PIL.Image.BILINEAR)


class PilTransform(Transform): _order, _valid = 11,False

class PIL_FLIP(PilTransform):
    def __init__(self,p): self.p = p
    def __call__(self,x): return x.transpose(random.randint(0,6)) if random.random() < self.p else x

class Enhancer(Transform): _order,_valid = 12, False

class BrEnhance(Enhancer):
    def __init__(self): self.en = ImageEnhance.Brightness
    def __call__(self,x): return self.en(x).enhance(random.uniform(0.5,1.5))

class ShEnhance(Enhancer):
    def __init__(self): self.en = ImageEnhance.Sharpness
    def __call__(self,x): return self.en(x).enhance(random.randint(-1,9))

class ConEnhance(Enhancer):
    def __init__(self): self.en = ImageEnhance.Contrast
    def __call__(self,x): return self.en(x).enhance(random.uniform(1,2))

class ColEnhance(Enhancer):
    def __init__(self): self.en = ImageEnhance.Color
    def __call__(self,x): return self.en(x).enhance(random.randint(1,3))

def np_to_float(x): 
    return torch.from_numpy(np.array(x, dtype=np.float32, copy=False)).permute(2,0,1).contiguous()/255.

np_to_float._order = 20

class PilRandomFlip(PilTransform):
    def __init__(self, p=0.5): self.p=p
    def __call__(self, x):
        return x.transpose(PIL.Image.FLIP_LEFT_RIGHT) if random.random()<self.p else x

class GeneralCrop(PilTransform):
    def __init__(self, size, crop_size=None, resample=PIL.Image.BILINEAR): 
        self.resample,self.size = resample,process_sz(size)
        self.crop_size = None if crop_size is None else process_sz(crop_size)
        
    def default_crop_size(self, w,h): return default_crop_size(w,h)

    def __call__(self, x):
        csize = self.default_crop_size(*x.size) if self.crop_size is None else self.crop_size
        return x.transform(self.size, PIL.Image.EXTENT, self.get_corners(*x.size, *csize), resample=self.resample)
    
    def get_corners(self, w, h): return (0,0,w,h)

class CenterCrop(GeneralCrop):
    def __init__(self, size, scale=1.14, resample=PIL.Image.BILINEAR):
        super().__init__(size, resample=resample)
        self.scale = scale
        
    def default_crop_size(self, w,h): return [w/self.scale,h/self.scale]
    
    def get_corners(self, w, h, wc, hc):
        return ((w-wc)//2, (h-hc)//2, (w-wc)//2+wc, (h-hc)//2+hc)

def process_sz(sz):
    sz = listify(sz)
    return tuple(sz if len(sz)==2 else [sz[0],sz[0]])



def default_crop_size(w,h): return [w,w] if w < h else [h,h]

class RandomResizedCrop(GeneralCrop):
    def __init__(self, size, scale=(0.08,1.0), ratio=(3./4., 4./3.), resample=PIL.Image.BILINEAR):
        super().__init__(size, resample=resample)
        self.scale,self.ratio = scale,ratio
    
    def get_corners(self, w, h, wc, hc):
        area = w*h
        #Tries 10 times to get a proper crop inside the image.
        for attempt in range(10):
            area = random.uniform(*self.scale) * area
            ratio = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
            new_w = int(round(math.sqrt(area * ratio)))
            new_h = int(round(math.sqrt(area / ratio)))
            if new_w <= w and new_h <= h:
                left = random.randint(0, w - new_w)
                top  = random.randint(0, h - new_h)
                return (left, top, left + new_w, top + new_h)
        
        # Fallback to squish
        if   w/h < self.ratio[0]: size = (w, int(w/self.ratio[0]))
        elif w/h > self.ratio[1]: size = (int(h*self.ratio[1]), h)
        else:                     size = (w, h)
        return ((w-size[0])//2, (h-size[1])//2, (w+size[0])//2, (h+size[1])//2)