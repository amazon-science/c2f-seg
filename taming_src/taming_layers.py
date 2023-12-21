# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x, mask=None):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = plm_conv(self.conv, x, mask, kernel_size=3, stride=2, padding=0)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                                     stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                                    stride=1, padding=0)

    def forward(self, x, mask=None):  
        # Resblock encoder decoder共用，这里如果是Decoder，mask就默认是None，那么plm_conv等价于conv
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = plm_conv(self.conv1, h, mask=mask, kernel_size=3, stride=1, padding=1)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = plm_conv(self.conv2, h, mask=mask, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = plm_conv(self.conv_shortcut, x, mask=mask, kernel_size=3, stride=1, padding=1)
            else:
                x = plm_conv(self.nin_shortcut, x, mask=mask, kernel_size=1, stride=1, padding=0)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, mask=None):
        # Encoder的attention需要利用mask做保密措施
        # mask:[b,1,h,w]
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        if mask is not None:
            mask_ = mask.reshape(b, 1, h * w).repeat(1, h * w, 1)  # b,1,hw->b,hw,hw
            w_masked = w_.masked_fill(mask_ == 1, -1e10) # 可能出现全1，所以不能用-inf
            w_masked = torch.softmax(w_masked, dim=2)  # 被mask区域为0
        else:
            w_masked = None
        w_ = torch.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)
        if mask is not None:
            # h_mask表示非mask区域的特征(他们不能看到被mask区域，所以是h_mask)
            # h_是mask区域，他们没有感受野限制，所以延用之前的特征
            w_masked = w_masked.permute(0, 2, 1)
            h_masked = torch.bmm(v, w_masked)  # b,c,hw
            h_masked = h_masked.reshape(b, c, h, w)
            h_ = h_masked * (1 - mask) + h_ * mask  # 根据mask将其合并

        h_ = self.proj_out(h_)

        return x + h_


def plm_conv(conv, x, mask=None, kernel_size=3, stride=1, padding=1):
    if mask is None:
        return conv(x)
    else:
        if stride > 1 and padding == 0:
            pad = (0, 1, 0, 1)
            mask = torch.nn.functional.pad(mask, pad, mode="constant", value=0)
        f1 = conv(x)  # original feature
        f2 = conv(x * (1 - mask))  # masked feature
        # 计算出可能泄露mask区域信息的范围:leak_area
        conv_weights = torch.ones(1, 1, kernel_size, kernel_size).to(dtype=x.dtype, device=x.device)
        leak_area = F.conv2d(mask, conv_weights, stride=stride, padding=padding)
        if stride > 1:
            mask = F.max_pool2d(mask, kernel_size=stride, stride=stride)
        leak_area[leak_area > 0] = 1
        leak_area = torch.clamp(leak_area - mask, 0, 1)  # leak_area减去
        # leak区域用 masked feature
        out = f1 * (1 - leak_area) + f2 * leak_area

        return out


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        ch = config['ch']
        num_res_blocks = config['num_res_blocks']
        # attn_resolutions = config['attn_resolutions']
        dropout = 0.0
        resamp_with_conv = True
        in_channels = config['in_channels']
        resolution_h = config['resolution_h']
        resolution_w = config['resolution_w']
        z_channels = config['z_channels']
        double_z = config['double_z']
        ch_mult = config['ch_mult']

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        # self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res_h = resolution_h
        curr_res_w = resolution_w
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            # attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                # if curr_res_h in attn_resolutions:
                #     attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            # down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res_h = curr_res_h // 2
                curr_res_w = curr_res_w // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        # self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, mask=None):  # 兼容之前的模型，mask默认为None   encoder

        # downsampling
        hs = [plm_conv(self.conv_in, x, mask=mask, kernel_size=3, stride=1, padding=1)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], mask)
                # if len(self.down[i_level].attn) > 0:
                    # h = self.down[i_level].attn[i_block](h, mask)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1], mask))
                if mask is not None:  # 保持mask和feature同shape
                    mask = F.max_pool2d(mask, kernel_size=int(mask.shape[2] / hs[-1].shape[2]),
                                        stride=int(mask.shape[2] / hs[-1].shape[2]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, mask)
        # h = self.mid.attn_1(h, mask)
        h = self.mid.block_2(h, mask)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = plm_conv(self.conv_out, h, mask=mask, kernel_size=3, stride=1, padding=1)
        return h


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        ch = config['ch']
        out_ch = config['out_ch']
        num_res_blocks = config['num_res_blocks']
        # attn_resolutions = config['attn_resolutions']
        dropout = 0.0
        resamp_with_conv = True
        in_channels = config['in_channels']
        resolution_h = config['resolution_h']
        resolution_w = config['resolution_w']
        z_channels = config['z_channels']
        double_z = config['double_z']
        ch_mult = config['ch_mult']
        give_pre_end = False

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        # self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res_h = resolution_h // 2 ** (self.num_resolutions - 1)
        curr_res_w = resolution_w // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res_h, curr_res_w)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        # self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            # attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                # if curr_res_h in attn_resolutions:
                    # attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            # up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res_h = curr_res_h * 2
                curr_res_w = curr_res_w * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):  # Decoder就都默认为None了
        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        # h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                # if len(self.up[i_level].attn) > 0:
                    # h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
