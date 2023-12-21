import os
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms

from taming_src.taming_models import VQModel
from src.video_component import MaskedTransformer, Resnet_Encoder, Refine_Module
from src.loss import VGG19, PerceptualLoss
from utils.pytorch_optimization import AdamW, get_linear_schedule_with_warmup
from utils.utils import torch_show_all_params, torch_init_model
from utils.utils import Config
from utils.evaluation import video_iou
from utils.loss import CrossEntropyLoss


class C2F_Seg(nn.Module):
    def __init__(self, config, g_path, mode, logger=None, save_eval_dict={}):
        super(C2F_Seg, self).__init__()
        self.config = config
        self.iteration = 0
        self.sample_iter = 0
        self.name = config.model_type
        # load g model for mask
        self.g_config = Config(os.path.join(g_path, 'vqgan_{}.yml'.format(config.dataset)))
        self.g_path = os.path.join(g_path, self.g_config.model_type)

        self.root_path = config.path
        self.transformer_path = os.path.join(config.path, self.name)
        # self.refine_path = os.path.join(config.path, "Refine")
        self.trans_size = config.trans_size
        self.mode = mode
        self.save_eval_dict = save_eval_dict

        self.eps = 1e-6
        self.train_sample_iters = config.train_sample_iters
        
        self.g_model = VQModel(self.g_config).to(config.device)
        self.img_encoder = Resnet_Encoder().to(config.device)
        self.refine_module = Refine_Module().to(config.device)
        self.transformer = MaskedTransformer(config).to(config.device)
        self.g_model.eval()

        self.refine_criterion = nn.BCELoss()
        self.criterion = CrossEntropyLoss(num_classes=config.vocab_size+1, device=config.device)

        if config.train_with_dec:
            if not config.gumbel_softmax:
                self.temperature = nn.Parameter(torch.tensor([config.tp], dtype=torch.float32),
                                                requires_grad=True).to(config.device)
            if config.use_vgg:
                vgg = VGG19(pretrained=True, vgg_norm=config.vgg_norm).to(config.device)
                vgg.eval()
                reduction = 'mean' if config.balanced_loss is False else 'none'
                self.perceptual_loss = PerceptualLoss(vgg, weights=config.vgg_weights,
                                                      reduction=reduction).to(config.device)
        else:
            self.perceptual_loss = None
    
        if config.init_gpt_with_vqvae:
            self.transformer.z_emb.weight = self.g_model.quantize.embedding.weight

        if logger is not None:
            logger.info('Gen Parameters:{}'.format(torch_show_all_params(self.g_model)))
            logger.info('Transformer Parameters:{}'.format(torch_show_all_params(self.transformer)))
        else:
            print('Gen Parameters:{}'.format(torch_show_all_params(self.g_model)))
            print('Transformer Parameters:{}'.format(torch_show_all_params(self.transformer)))

        # loss
        no_decay = ['bias', 'ln1.bias', 'ln1.weight', 'ln2.bias', 'ln2.weight']
        ignored_param = ['z_emb.weight', 'c_emb.weight']
        param_optimizer = self.transformer.named_parameters()
        param_optimizer_encoder = self.img_encoder.named_parameters()
        param_optimizer_refine= self.refine_module.named_parameters()
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any([nd in n for nd in no_decay])],
            'weight_decay': config.weight_decay},
            {'params': [p for n, p in param_optimizer if any([nd in n for nd in no_decay])],
            'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer_encoder], 'weight_decay': config.weight_decay},
            {'params': [p for n, p in param_optimizer_refine], 'weight_decay': config.weight_decay},
        ]

        self.opt = AdamW(params=optimizer_parameters,
                         lr=float(config.lr), betas=(config.beta1, config.beta2))
        self.sche = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_iters,
                                                    num_training_steps=config.max_iters)

        self.rank = dist.get_rank()
        self.gamma = self.gamma_func(mode=config.gamma_mode)
        self.mask_token_idx = config.vocab_size
        self.choice_temperature = 4.5
        self.Image_W = config.Image_W
        self.Image_H = config.Image_H
        self.patch_W = config.patch_W
        self.patch_H = config.patch_H

    @torch.no_grad()
    def encode_to_z(self, x, mask=None):
        if len(x.size())==5:
            x = x[0]
        x = x.permute((1,0,2,3))
        quant_z, _, info = self.g_model.encode(x.float(), mask)  # [B,D,H,W]
        indices = info[2].view(quant_z.shape[0], -1)  # [B, L]
        return quant_z, indices

    def data_augmentation(self, mask):
        w = random.randint(5, 11)
        h = random.randint(5, 11)
        rdv = random.random()
        n_repeat = random.randint(1, 3)
        max_pool = nn.MaxPool2d(kernel_size=(w, h), stride=1, padding=(w//2, h//2))
        if rdv < 0.3:
            for i in range(n_repeat):
                mask = max_pool(mask)
        elif rdv >=0.3 and rdv < 0.6:
            for i in range(n_repeat):
                mask = -max_pool(-mask)
        else:
            mask = mask
        return mask
    
    def get_attn_map(self, feature, guidance):
        guidance = F.interpolate(guidance, scale_factor=(1/16))
        b,c,h,w = guidance.shape
        q = torch.flatten(guidance, start_dim=2)
        v = torch.flatten(feature, start_dim=2)

        k = v * q
        k = k.sum(dim=-1, keepdim=True) / (q.sum(dim=-1, keepdim=True) + 1e-6)
        attn = (k.transpose(-2, -1) @  v) / 1
        attn = F.softmax(attn, dim=-1)
        attn = attn.reshape(b, c, h, w)
        return attn

    def get_losses(self, meta):
        self.iteration += 1
        z_loss = 0
        img_feat = self.img_encoder(meta['img_crop'].squeeze().permute((0,3,1,2)).to(torch.float32))
        _, src_indices = self.encode_to_z(meta['vm_crop'])
        _, tgt_indices = self.encode_to_z(meta['fm_crop'])
        bhwc = (_.shape[0], _.shape[2], _.shape[3], _.shape[1])
        r = np.maximum(self.gamma(np.random.uniform()), self.config.min_mask_rate)
        r = math.floor(r * tgt_indices.shape[1])
        sample = torch.rand(tgt_indices.shape, device=tgt_indices.device).topk(r, dim=1).indices
        random_mask = torch.zeros(tgt_indices.shape, dtype=torch.bool, device=tgt_indices.device)
        random_mask.scatter_(dim=1, index=sample, value=True) # [B, L]
        # concat mask
        mask = random_mask
        masked_indices = self.mask_token_idx * torch.ones_like(tgt_indices, device=tgt_indices.device) # [B, L]
        z_indices = (~mask) * tgt_indices + mask * masked_indices # [B, L]

        logits_z = self.transformer(img_feat[-1], src_indices, z_indices, mask=None)        
        target = tgt_indices
        z_loss = self.criterion(logits_z.view(-1, logits_z.size(-1)), target.view(-1))

        with torch.no_grad():
            logits_z = logits_z[..., :-1]
            logits_z = self.top_k_logits(logits_z, k=5)
            probs = F.softmax(logits_z, dim=-1)
            seq_ids = torch.distributions.categorical.Categorical(probs=probs).sample() # [B, L]
            quant_z = self.g_model.quantize.get_codebook_entry(seq_ids.reshape(-1), shape=bhwc)
            pred_fm_crop = self.g_model.decode(quant_z)
            pred_fm_crop = pred_fm_crop.mean(dim=1, keepdim=True)
            pred_fm_crop = torch.clamp(pred_fm_crop, min=0, max=1)

        pred_vm_crop, pred_fm_crop = self.refine_module(img_feat, pred_fm_crop.detach())
        pred_vm_crop = F.interpolate(pred_vm_crop, size=(256, 256), mode="nearest")
        pred_vm_crop = torch.sigmoid(pred_vm_crop)

        loss_vm = self.refine_criterion(pred_vm_crop, meta['vm_crop_gt'].transpose(1,0))

        pred_fm_crop = F.interpolate(pred_fm_crop, size=(256, 256), mode="nearest")
        pred_fm_crop = torch.sigmoid(pred_fm_crop)
        loss_fm = self.refine_criterion(pred_fm_crop, meta['fm_crop'].transpose(1,0))

        logs = [
            ("z_loss", z_loss.item()),
            ("loss_vm", loss_vm.item()),
            ("loss_fm", loss_fm.item()),
        ]
        return z_loss, loss_vm+loss_fm, logs
    
    def align_raw_size(self, full_mask, obj_position, vm_pad, meta):
        vm_np_crop = meta["vm_no_crop"].squeeze()
        H, W = vm_np_crop.shape[-2], vm_np_crop.shape[-1]
        bz, seq_len = full_mask.shape[:2]
        new_full_mask = torch.zeros((bz, seq_len, H, W)).to(torch.float32).cuda()
        if len(vm_pad.shape)==3:
            vm_pad = vm_pad[0]
            obj_position = obj_position[0]
        for b in range(bz):
            paddings = vm_pad[b]
            position = obj_position[b]
            new_fm = full_mask[
                b, :,
                :-int(paddings[0]) if int(paddings[0]) !=0 else None,
                :-int(paddings[1]) if int(paddings[1]) !=0 else None
            ]
            vx_min = int(position[0])
            vx_max = min(H, int(position[1])+1)
            vy_min = int(position[2])
            vy_max = min(W, int(position[3])+1)
            resize = transforms.Resize([vx_max-vx_min, vy_max-vy_min])
            try:
                new_fm = resize(new_fm)
                new_full_mask[b, :, vx_min:vx_max, vy_min:vy_max] = new_fm[0]
            except:
                new_fm = new_fm
        return new_full_mask

    def loss_and_evaluation(self, pred_fm, meta):
        loss_eval = {}
        pred_fm = pred_fm.squeeze()
        loss_mask = meta["loss_mask"].squeeze().to(pred_fm.device)
        counts = meta["counts"].reshape(-1).to(pred_fm.device)
        fm_no_crop = meta["fm_no_crop"].squeeze()
        vm_no_crop = meta["vm_no_crop"].squeeze()
        pred_fm = (pred_fm > 0.5).to(torch.int64)
        full_iou = video_iou(pred_fm, fm_no_crop)
        occ_iou = video_iou(pred_fm-vm_no_crop, fm_no_crop-vm_no_crop)
        m_full_iou = (counts * full_iou).sum() / counts.sum()
        m_occ_iou = (counts * occ_iou).sum() / counts.sum()
        loss_eval["iou"] = m_full_iou
        loss_eval["invisible_iou_"] = m_occ_iou
        loss_eval["iou_count"] = torch.Tensor([1]).cuda()
        loss_eval["occ_count"] = torch.Tensor([1]).cuda()
        # post-process
        pred_fm = pred_fm * (1 - loss_mask) + vm_no_crop * loss_mask
        pred_fm = (pred_fm > 0.5).to(torch.int64)
        full_iou = video_iou(pred_fm, fm_no_crop)
        occ_iou = video_iou(pred_fm-vm_no_crop, fm_no_crop-vm_no_crop)
        m_full_iou = (counts * full_iou).sum() / counts.sum()
        m_occ_iou = (counts * occ_iou).sum() / counts.sum()
        loss_eval["iou_post"] = m_full_iou
        loss_eval["invisible_iou_post"] = m_occ_iou
        return loss_eval

    def backward(self, loss=None):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.sche.step()

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def batch_predict_maskgit(self, meta, iter, mode, temperature=1.0, T=3, start_iter=0):
        '''
        :param x:[B,3,H,W] image
        :param c:[b,X,H,W] condition
        :param mask: [1,1,H,W] mask
        '''
        self.sample_iter += 1
        img_feat = self.img_encoder(meta['img_crop'].squeeze().permute((0,3,1,2)).to(torch.float32))
        _, src_indices = self.encode_to_z(meta['vm_crop'])
        # _, tgt_indices = self.encode_to_z(meta['fm_crop'])
        bhwc = (_.shape[0], _.shape[2], _.shape[3], _.shape[1])

        masked_indices = self.mask_token_idx * torch.ones_like(src_indices, device=src_indices.device) # [B, L]
        unknown_number_in_the_beginning = torch.sum(masked_indices == self.mask_token_idx, dim=-1) # [B]

        gamma = self.gamma_func("cosine")
        cur_ids = masked_indices # [B, L]
        seq_out = []
        mask_out = []

        for t in range(start_iter, T):
            logits = self.transformer(img_feat[-1], src_indices, cur_ids, mask=None) # [B, L, N]
            logits = logits[..., :-1]
            logits = self.top_k_logits(logits, k=3)
            probs = F.softmax(logits, dim=-1)  # convert logits into probs [B, 256, vocab_size+1]
            sampled_ids = torch.distributions.categorical.Categorical(probs=probs).sample() # [B, L]

            unknown_map = (cur_ids == self.mask_token_idx)  # which tokens need to be sampled -> bool [B, 256]
            sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)  # replace all -1 with their samples and leave the others untouched [B, 256]
            seq_out.append(sampled_ids)
            mask_out.append(1. * unknown_map)

            ratio = 1. * (t + 1) / T  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)
            selected_probs = probs.gather(dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze(-1)

            selected_probs = torch.where(unknown_map, selected_probs, torch.Tensor([np.inf]).to(logits.device))  # ignore tokens which are already sampled [B, 256]
            
            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)  # floor(256 * 0.99) = 254 --> [254, 254, 254, 254, ....] (B x 1)
            mask_len = torch.maximum(torch.ones_like(mask_len), torch.minimum(torch.sum(unknown_map, dim=-1, keepdim=True) - 1, mask_len))

            # Adds noise for randomness
            masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=self.choice_temperature * (1. - ratio))
            # Masks tokens with lower confidence.
            cur_ids = torch.where(masking, self.mask_token_idx, sampled_ids) # [B, L]

        seq_ids = torch.stack(seq_out, dim=1) # [B, T, L]
        quant_z = self.g_model.quantize.get_codebook_entry(seq_ids[:,-1,:].reshape(-1), shape=bhwc)
        pred_fm_crop = self.g_model.decode(quant_z)
        pred_fm_crop = pred_fm_crop.mean(dim=1, keepdim=True)
        pred_fm_crop_old = torch.clamp(pred_fm_crop, min=0, max=1)

        pred_vm_crop, pred_fm_crop = self.refine_module(img_feat, pred_fm_crop_old)

        pred_vm_crop = F.interpolate(pred_vm_crop, size=(256, 256), mode="nearest")
        pred_vm_crop = torch.sigmoid(pred_vm_crop)
        loss_vm = self.refine_criterion(pred_vm_crop, meta['vm_crop'].transpose(1,0))
        # pred_vm_crop = (pred_vm_crop>=0.5).to(torch.float32)

        pred_fm_crop = F.interpolate(pred_fm_crop, size=(256, 256), mode="nearest")
        pred_fm_crop = torch.sigmoid(pred_fm_crop)
        loss_fm = self.refine_criterion(pred_fm_crop, meta['fm_crop'].transpose(1,0))
        # pred_fm_crop = (pred_fm_crop>=0.5).to(torch.float32)

        pred_fm_crop_old = self.align_raw_size(pred_fm_crop_old, meta['obj_position'], meta["vm_pad"], meta)
        # pred_vm = self.align_raw_size(pred_vm_crop, meta['obj_position'], meta["vm_pad"], meta)
        pred_fm = self.align_raw_size(pred_fm_crop, meta['obj_position'], meta["vm_pad"], meta)
        pred_fm = pred_fm + pred_fm_crop_old
        loss_eval = self.loss_and_evaluation(pred_fm, meta)
        loss_eval["loss_fm"] = loss_fm
        loss_eval["loss_vm"] = loss_vm
        return loss_eval

    def create_inputs_tokens_normal(self, num, device):
        self.num_latent_size = self.config['resolution'] // self.config['patch_size']
        blank_tokens = torch.ones((num, self.num_latent_size ** 2), device=device)
        masked_tokens = self.mask_token_idx * blank_tokens

        return masked_tokens.to(torch.int64)

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        elif mode == "log":
            return lambda r, total_unknown: - np.log2(r) / np.log2(total_unknown)
        else:
            raise NotImplementedError

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0):
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to(probs.device)
        sorted_confidence, _ = torch.sort(confidence, dim=-1) # from small to large
        # Obtains cut off threshold given the mask lengths.
        # cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        cut_off = sorted_confidence.gather(dim=-1, index=mask_len.to(torch.long))
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking
    
    def load(self, is_test=False, prefix=None):
        if prefix is not None:
            transformer_path = self.transformer_path + prefix + '.pth'
        else:
            transformer_path = self.transformer_path + '_last.pth'
        if self.config.restore or is_test:
            # transformer_path = '/home/ubuntu/AmodalVQGAN_pre/check_points/fish_amodal_transformer_gjx/GETransformer_210000.pth'
            if os.path.exists(transformer_path):
                print('Rank {} is loading {} Transformer...'.format(self.rank, transformer_path))
                data = torch.load(transformer_path, map_location="cpu")
                torch_init_model(self.transformer, transformer_path, 'model')
                torch_init_model(self.img_encoder, transformer_path, 'img_encoder')
                torch_init_model(self.refine_module, transformer_path, 'refine')

                if self.config.restore:
                    self.opt.load_state_dict(data['opt'])
                    # 空过sche
                    from tqdm import tqdm
                    for _ in tqdm(range(data['iteration']), desc='recover sche...'):
                        self.sche.step()
                self.iteration = data['iteration']
                self.sample_iter = data['sample_iter']
            else:
                print(transformer_path, 'not Found')
                raise FileNotFoundError

    def restore_from_stage1(self, prefix=None):
        if prefix is not None:
            g_path = self.g_path + prefix + '.pth'
        else:
            g_path = self.g_path + '_last.pth'
        if os.path.exists(g_path):
            print('Rank {} is loading {} G Mask ...'.format(self.rank, g_path))
            torch_init_model(self.g_model, g_path, 'g_model')
        else:
            print(g_path, 'not Found')
            raise FileNotFoundError
    
    def save(self, prefix=None):
        if prefix is not None:
            save_path = self.transformer_path + "_{}.pth".format(prefix)
        else:
            save_path = self.transformer_path + ".pth"

        print('\nsaving {} {}...\n'.format(self.name, prefix))
        torch.save({
            'iteration': self.iteration,
            'sample_iter': self.sample_iter,
            'model': self.transformer.state_dict(),
            'img_encoder': self.img_encoder.state_dict(),
            'refine': self.refine_module.state_dict(),
            'opt': self.opt.state_dict(),
        }, save_path)

        