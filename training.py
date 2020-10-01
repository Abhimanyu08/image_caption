from utils import *
from data_block import *
from transforms import *
from transformers import GPT2Tokenizer
from model import *
from learner import *
from callbacks import *
from optimizers import *
import torch
import torchvision.models as models
import json
import re

def get_caption_data(path, ann_path, tfms,image_folders, train_caption_file, valid_caption_file):
    il = ImageList.from_path(path, tfms = tfms, include = image_folders)
    train_ann = json.load(open(os.path.join(path, ann_path, train_caption_file)))
    valid_ann = json.load(open(os.path.join(path, ann_path, valid_caption_file)))
    dic = dict([(i['image_id'], i['caption']) for i in train_ann['annotations']])
    dic.update([(i['image_id'], i['caption']) for i in valid_ann['annotations']])
    return il, dic


il, cap_dict = get_caption_data(path = 'C:/Users/iamab/Downloads/coco',
                               ann_path= 'annotations_trainval2014/annotations',
                               tfms = [into_rgb, ResizeFixed(224), np_to_float],
                               image_folders = ['train_images2014', 'valid_images2014'],
                               train_caption_file = 'captions_train2014.json',
                               valid_caption_file = 'captions_val2014.json')


def parent_splitter(o, name): return True if o.parent.name == name else False

sl = SplitDataset.from_func(il, partial(parent_splitter, name = 'val2014'))

def get_captions(o,cap_dict):
    im_id = int(re.findall(r'0*(\d+).jpg', o.name)[0])
    return cap_dict[im_id]

ll = label_train_valid(sl, func = partial(get_captions, cap_dict = cap_dict),
                       procy = Tokprocessor(GPT2Tokenizer.from_pretrained('gpt2')))

def pad_collate(samples, pad_idx,flip = False):
    images = torch.stack([s[0] for s in samples])
    if flip:
        captions = [s[1].contiguous() for s in samples]
        for s in captions:
            idx = next(j for j,n in enumerate(s) if n == pad_idx)
            cut = s[:idx].flip(0)
            s[:idx] = cut
    res = torch.stack([s[1] for s in samples])
            
    return (images, res.long()), torch.cat((res[:,1:], torch.zeros(res.size(0),1).int()+pad_idx), dim=1).long()

def cap_databunchify(sd, train_bs = None,valid_bs = None,**kwargs):
    return DataBunch(*get_cap_dls(sd, train_bs, valid_bs, collate_fn = pad_collate))

SplitDataset.cap_databunch = cap_databunchify

data = ll.cap_databunch(train_bs=8, num_workers = 2)


visual_model = models.resnet50(pretrained= True, progress= True)

cap_model = get_caption_model(data,num_visual_features=2048,
                              visual_head=visual_model, N = 4, activation = 'gelu')

optimizer = partial(StatefulOptimizer, 
                    steppers = [momentum_step, weight_decay,lookahead_step],
                   stats = [AverageGrad(), Step(), LookAheadStat()],
                   lr = 0.02, wd = 1e-04)

class GradZeroCallback(Callback):
    def begin_fit(self):
        for p in self.model.visual_backbone.parameters():
            p.requires_grad_(False)

cbfs = [CudaCapCallback, partial(AvgStatsCallback, cap_accuracy, AvgStatsCap), ProgressCallback, 
        normalise_callback(data, normalise_func= cap_normalise), NewRecorderCallback, WandbCallback, 
        GradZeroCallback]

wandb.init(name = 'vb_frozen', project = 'yo',entity = 'a_bhimanyu', 
          dir = 'C:/Users/iamab/Google Drive')

loss = FlatLoss(nn.CrossEntropyLoss(ignore_index = cap_model.pad_tok_id))



learn = Learner(cap_model, data, loss, optimizer, cb_funcs= cbfs)

learn.fit(2)