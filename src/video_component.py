import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from torchvision.models import resnet18, resnet50

logger = logging.getLogger(__name__)

class base_resnet(nn.Module):
    def __init__(self):
        super(base_resnet, self).__init__()
        self.model = resnet50(pretrained=True)
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        features = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        features.append(x)
        x = self.model.layer2(x)
        features.append(x)
        x = self.model.layer3(x)
        features.append(x)
        x = self.model.layer4(x)
        features.append(x)
        return features

class Resnet_Encoder(nn.Module):
    def __init__(self):
        super(Resnet_Encoder, self).__init__()
        self.encoder = base_resnet()

    def forward(self, img):
        features = self.encoder(img)
        return features
    
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

def window_partition(x, window_shape):
    """
    Modified from: https://github.com/microsoft/Swin-Transformer/blob/d19503d7fbed704792a5e5a3a5ee36f9357d26c1/models/swin_transformer_v2.py#L35
    Args:
        x: (B, H, W, C)
        window_size: (b, h, w)
    Returns:
        windows: 
    """
    b,h,w = window_shape
    B, H, W, C = x.shape
    x = x.view(B//b, b, H//h, h, W//w, w, C)
    windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, b*h*w, C)
    return windows

def window_reverse(windows, window_shape, H, W):
    """
    Modified from: https://github.com/microsoft/Swin-Transformer/blob/d19503d7fbed704792a5e5a3a5ee36f9357d26c1/models/swin_transformer_v2.py#L50
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_shape: (b, h, w)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    b, h, w = window_shape
    C = windows.shape[-1]
    B = int(windows.shape[0] / (H * W) * (b*h*w))
    x = windows.view(B//b, H // h, W // w, b, h, w, C)
    x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(-1, H*W, C)
    return x

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # mask:[B,1,L,L]
        att = att.masked_fill(mask == 0, float('-inf'))

        if x.dtype == torch.float16:
            att = att.to(torch.float32)
            fp16 = True
        else:
            fp16 = False
        att = F.softmax(att, dim=-1)
        if fp16:
            att = att.to(torch.float16)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block like the one in MaskVit by FeiFei Li """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.attn1 = CausalSelfAttention(config)
        self.attn2 = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            # nn.GELU(),  # nice, GELU is not valid in torch<1.6
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        
    def forward(self, x, window_size=(16, 4, 4), mask=None):
        sq_len, tokens, d = x.shape
        # spatial block
        x = x + self.attn1(self.ln1(x), mask)
        # spatial-temporal block
        x = x.reshape((sq_len, 16, 16, d))
        windows = window_partition(x, window_size)
        B, T, C = windows.shape
        mask_t = torch.ones(B, 1, T, T).cuda()
        windows = windows + self.attn2(self.ln2(windows), mask_t)
        x = window_reverse(windows, window_size, 16, 16)
        x = x + self.mlp(self.ln3(x))
        return x

class Transformer_Prediction(nn.Module):
    def __init__(self, config):
        super(Transformer_Prediction, self).__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        self.gelu = GELU()
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = self.ln1(x)
        x = self.dense(x)
        x = self.gelu(x)
        x = self.ln2(x)
        return x
    
class MaskedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_dim = config.n_embd
        num_embed = config.vocab_size+1
        self.conv_in = torch.nn.Conv2d(2048, embedding_dim//2, 3, padding=1)
        # z_embedding
        self.c_emb = nn.Embedding(num_embed, embedding_dim//4)
        self.z_emb = nn.Embedding(num_embed, embedding_dim//4)
        # posotion embedding
        self.pos_emb = nn.Embedding(config.sequence_length, embedding_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.dec = Transformer_Prediction(config)
        # z dec and m dec
        self.m_dec = nn.Linear(embedding_dim, num_embed, bias=False)
        self.m_bias = nn.Parameter(torch.zeros(num_embed))

        self.sequence_length = config.sequence_length
        self.apply(self._init_weights)
        self.config = config
        self.window_len = int(self.config.window_length)

    def forward(self, img_feat, c_idx, z_idx, window_size=(12, 4, 4), mask=None):
        # img_feat: [B, 2048, 16, 16]
        # attn_map: [B, 1,    16, 16]
        i_embeddings = self.conv_in(img_feat) # [B, 768//2-1, 16, 16]
        i_embeddings = i_embeddings.flatten(2).transpose(-2, -1)
        # c and z embedding
        c_embeddings = self.c_emb(c_idx)    # [B, 256, D//4]
        z_embeddings = self.z_emb(z_idx)    # [B, 256, D//4]
        token_embeddings = torch.cat([i_embeddings, c_embeddings, z_embeddings], dim=2) # [B, 256, D]
        # add positional embeddings
        n_tokens = token_embeddings.shape[1] # 16 * 16
        position_ids = torch.arange(n_tokens, dtype=torch.long, device=z_idx.device)
        position_ids = position_ids.unsqueeze(0).repeat(z_idx.shape[0], 1) # [B, 256, 1]
        position_embeddings = self.pos_emb(position_ids)                   # [B, 256, D]

        x = self.drop(token_embeddings + position_embeddings)

        batch_size = token_embeddings.shape[0]
        mask = torch.ones(batch_size, 1, n_tokens, n_tokens).cuda()
        window_size = (self.window_len, 4, 4)

        for block in self.blocks:
            x = block(x, window_size=window_size, mask=mask)
            x = torch.roll(x, self.window_len//2, 0)

        total_shift_size = (self.window_len//2) * len(self.blocks)
        x = torch.roll(x, batch_size - total_shift_size%batch_size, 0)

        x = self.dec(x)
        logits_m = self.m_dec(x) + self.m_bias
        
        return logits_m

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class Refine_Module(nn.Module):
    def __init__(self):
        super(Refine_Module, self).__init__()
        # self.encoder = base_resnet()
        dim = 256 + 2
        self.conv_adapter = torch.nn.Conv2d(2048, 2048, 1)
        self.conv_in = torch.nn.Conv2d(2048, 256, 3, padding=1)
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(dim)

        self.lay2 = torch.nn.Conv2d(dim, 128, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)

        self.lay3 = torch.nn.Conv2d(128, 64, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.adapter1 = torch.nn.Conv2d(1024, 128, 1)

        # visible mask branch
        self.lay4_vm = torch.nn.Conv2d(64, 32, 3, padding=1)
        self.bn4_vm = torch.nn.BatchNorm2d(32)
        self.lay5_vm = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.bn5_vm = torch.nn.BatchNorm2d(16)
        self.adapter2_vm = torch.nn.Conv2d(512, 64, 1)
        self.adapter3_vm = torch.nn.Conv2d(256, 32, 1)
        self.out_lay_vm = torch.nn.Conv2d(16, 1, 3, padding=1)
    
        # full mask branch
        self.lay4_am = torch.nn.Conv2d(64, 32, 3, padding=1)
        self.bn4_am = torch.nn.BatchNorm2d(32)
        self.lay5_am = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.bn5_am = torch.nn.BatchNorm2d(16)
        self.adapter2_am = torch.nn.Conv2d(512, 64, 1)
        self.adapter3_am = torch.nn.Conv2d(256, 32, 1)
        self.out_lay_am = torch.nn.Conv2d(16, 1, 3, padding=1)
    
    def get_attn_map(self, feature, guidance):
        b,c,h,w = guidance.shape
        q = torch.flatten(guidance, start_dim=2)
        v = torch.flatten(feature, start_dim=2)

        k = v * q
        k = k.sum(dim=-1, keepdim=True) / (q.sum(dim=-1, keepdim=True) + 1e-6)
        attn = (k.transpose(-2, -1) @  v) / 1
        attn = F.softmax(attn, dim=-1)
        attn = attn.reshape(b, c, h, w)
        return attn
    
    def forward(self, features, coarse_mask):
        # features:    [B, 2048, 16,   16]
        # attn_map:    [B, 1,    16,   16]
        # coarse_mask: [B, 1,    256, 256]
        feat = self.conv_adapter(features[-1])
        coarse_mask = F.interpolate(coarse_mask, scale_factor=(1/16))
        attn_map = self.get_attn_map(feat, coarse_mask)
        x = self.conv_in(feat)
        x = torch.cat((x, attn_map, coarse_mask), dim=1)
        x = F.relu(self.bn1(self.lay1(x)))
        x = F.relu(self.bn2(self.lay2(x)))
        
        cur_feat = self.adapter1(features[-2])
        x = cur_feat + x
        x = F.interpolate(x, size=(32, 32), mode="nearest")
        x = F.relu(self.bn3(self.lay3(x)))

        # TODO: visible mask branch
        cur_feat_vm = self.adapter2_vm(features[-3])
        x_vm = cur_feat_vm + x
        x_vm = F.interpolate(x_vm, size=(64, 64), mode="nearest")
        x_vm = F.relu(self.bn4_vm(self.lay4_vm(x_vm)))

        cur_feat_vm = self.adapter3_vm(features[-4])
        x_vm = cur_feat_vm + x_vm
        x_vm = F.interpolate(x_vm, size=(128, 128), mode="nearest")
        x_vm = F.relu(self.bn5_vm(self.lay5_vm(x_vm)))
        
        x_vm = self.out_lay_vm(x_vm)

        # TODO: full mask branch
        cur_feat_am = self.adapter2_am(features[-3])
        x_am = cur_feat_am + x
        x_am = F.interpolate(x_am, size=(64, 64), mode="nearest")
        x_am = F.relu(self.bn4_am(self.lay4_am(x_am)))

        cur_feat_am = self.adapter3_am(features[-4])
        x_am = cur_feat_am + x_am
        x_am = F.interpolate(x_am, size=(128, 128), mode="nearest")
        x_am = F.relu(self.bn5_am(self.lay5_am(x_am)))
        
        x_am = self.out_lay_am(x_am)

        return x_vm, x_am