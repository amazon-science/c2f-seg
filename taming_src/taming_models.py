import torch
import torch.nn as nn
import os
from taming_src.taming_layers import Encoder, Decoder
from taming_src.vqperceptual import VQLPIPSWithDiscriminator
from src.quantize import VectorQuantizer
from utils.utils import torch_init_model
import torch.nn.functional as F


class VQModel(nn.Module):
    def __init__(self, config):
        super(VQModel, self).__init__()
        self.config = config
        self.iteration = 0
        self.name = config.model_type
        self.m_path = os.path.join(config.path, self.name)
        self.eps = 1e-6

        self.ddconfig = config.model['params']['ddconfig']
        n_embed = config.model['params']['n_embed']
        embed_dim = config.model['params']['embed_dim']
        
        self.encoder = Encoder(self.ddconfig).to(config.device)
        self.decoder = Decoder(self.ddconfig).to(config.device)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25).to(config.device).to(config.device)
        self.quant_conv = torch.nn.Conv2d(self.ddconfig["z_channels"], embed_dim, 1).to(config.device)
        # self.quant_proj = torch.nn.Linear(self.ddconfig["z_channels"], embed_dim).to(config.device)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, self.ddconfig["z_channels"], 1).to(config.device)
        # self.pose_quant_proj = torch.nn.Linear(embed_dim, self.ddconfig["z_channels"]).to(config.device)

    def encode(self, x, mask=None):
        h = self.encoder(x) # dim=256
        h = self.quant_conv(h) # dim=256
        if mask is not None:
            mask = F.max_pool2d(mask, kernel_size=int(mask.shape[2] / h.shape[2]),
                                stride=int(mask.shape[2] / h.shape[2]))
            quant = quant * mask + h * (1 - mask)
        quant, emb_loss, info = self.quantize(h, mask)
        
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant) # dim: 256
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, x, mask=None):
        quant, diff, _ = self.encode(x, mask) # quant dim: 256

        dec = self.decode(quant)
        return dec, diff

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def restore(self, ckpt_file, g_opt=None, d_opt=None):
        torch_init_model(self, ckpt_file, "state_dict")
        saving = torch.load(ckpt_file, map_location='cpu')
        if 'optimizer_states' in saving and g_opt is not None and d_opt is not None:
            opt_state = saving['optimizer_states']
            g_opt.load_state_dict(opt_state[0])
            d_opt.load_state_dict(opt_state[1])
        print(f"Restored from {ckpt_file}")
        return g_opt, d_opt

    def save(self, prefix=None, g_opt=None, d_opt=None):
        if prefix is not None:
            save_path = self.m_path + "_{}.pth".format(prefix)
        else:
            save_path = self.m_path + ".pth"

        print('\nsaving {} {}...\n'.format(self.name, prefix))
        all_saving = {'state_dict': self.state_dict(),
                      'optimizer_states': [g_opt.state_dict(), d_opt.state_dict()]}
        torch.save(all_saving, save_path)
