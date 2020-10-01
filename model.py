from torch import nn
from transformer import Xcoder, get_padding_mask, attention_mask, WordEmbedding, PositionalEncoding
from utils import partial

class CaptionModel(nn.Module):
    def __init__(self, visual_model, num_visual_features, textual_features, vocab_size,pad_token_id,
                max_len = 49, encoding_drop = 0.1, N = 6, heads = 8 , 
                attn_drop = 0.1, ff_drop = 0.1,d_ff = 2048, activation = 'GELU'):
        super().__init__()
        self.visual_backbone = visual_model
        self.th = Xcoder(True, N, textual_features, h = heads,d_ff=d_ff, ff_drop=ff_drop, attn_drop=attn_drop,
                         activation= activation)
        self.visual_features = []
        self.lin_projection = nn.Linear(num_visual_features, textual_features)
        self.embed = WordEmbedding(vocab_size, textual_features, padding_index=pad_token_id)
        self.pos_enc = PositionalEncoding(textual_features, max_len, encoding_drop)
        self._register_hook(self.visual_backbone, partial(self.hook_function, self.visual_features))
        self.lin_out = nn.Linear(textual_features, vocab_size)
        self.lin_out.weight = self.embed.emb.weight
        self.pad_tok_id = pad_token_id
        
    def forward(self,inp):
        image,caption = inp
        self.visual_backbone(image)
        vf = self.visual_features[0]
        self.visual_features.clear()
        vf = vf.view(vf.size(0), vf.size(1), -1).permute(0,2,1)
        vf = self.lin_projection(vf)
        sub_mask = get_padding_mask(caption,pad_idx = self.pad_tok_id) + attention_mask(caption)
        pad_mask = get_padding_mask(caption,pad_idx = self.pad_tok_id)
        caption = self.pos_enc(self.embed(caption))
        out = self.th(caption, vf,vf,mask_in = sub_mask, mask_out  = pad_mask)
        return self.lin_out(out)
        
    @staticmethod    
    def _register_hook(model,func):
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Conv2d):
                m.register_forward_hook(func)
                break
                
    @staticmethod            
    def hook_function(obj,mod,inp,out):
        obj.append(out)
                
    

def get_caption_model(data, visual_head, num_visual_features = 512, num_textual_features = 512,**kwargs):
    tok = data.train_ds.py.tokenizer
    ml = data.train_ds.py.ml
    return CaptionModel(visual_head, num_visual_features, num_textual_features,tok.vocab_size+3,
                       tok.pad_token_id, ml, **kwargs)
    