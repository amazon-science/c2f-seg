import time
import sys
import numpy as np
from PIL import Image, ImageDraw
import math
import yaml
import os
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch
import torchvision
import cv2

import random
import torch.distributed as dist

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def reduce_tensors(tensor):
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor)
    return reduced_tensor

# def average_gradients(model):
#     print(model)
#     """ average gradients """
#     for param in model.parameters():
#         if param.requires_grad:
#             dist.all_reduce(param.grad.data)

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
        
# irregular mask generation
def irregular_mask(H, W, min_width=12, max_width=40, min_brush=2, max_brush=5,
                   min_num_vertex=4, max_num_vertex=12):
    mean_angle = 2 * math.pi / 5
    angle_range = 2 * math.pi / 15
    average_radius = math.sqrt(H * H + W * W) / 8
    mask = Image.new('L', (W, H), 0)

    for _ in range(np.random.randint(min_brush, max_brush + 1)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius // 2),
                0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width // 2,
                          v[1] - width // 2,
                          v[0] + width // 2,
                          v[1] + width // 2),
                         fill=1)

    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.float32)
    # mask = np.reshape(mask, (H, W, 1))
    return mask


class Config(object):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.load(self._yaml, Loader=yaml.SafeLoader)
            self._dict['path'] = os.path.dirname(config_path)

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]

        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')


class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, max_iters=None, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.max_iters = max_iters
        self.iters = 0
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            # if self.target is not None and current < self.target:
            if self.max_iters is None or self.iters < self.max_iters:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    if 'lr' in k:
                        info += ' %.3e' % self._values[k]
                    else:
                        info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.iters += 1
        self.update(self._seen_so_far + n, values)


def torch_show_all_params(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    return k


def to_cuda(meta, device):
    for k in meta:
        if  meta[k] is not None:
            meta[k] = meta[k].to(device)
    return meta


def get_lr_schedule_with_steps(decay_type, optimizer, drop_steps=None, gamma=None, total_steps=None):
    def lr_lambda(current_step):
        if decay_type == 'fix':
            return 1.0
        elif decay_type == 'linear':
            return 1.0 * (current_step / total_steps)
        elif decay_type == 'cos':
            return 1.0 * (math.cos((current_step / total_steps) * math.pi) + 1) / 2
        elif decay_type == 'milestone':
            return 1.0 * math.pow(gamma, int(current_step / drop_steps))
        else:
            raise NotImplementedError

    return LambdaLR(optimizer, lr_lambda)


def stitch_images(inputs, outputs, img_per_row=2, mode="L"):
    gap = 5
    columns = len(outputs) + 1

    height, width = inputs[0][:, :, 0].shape
    img = Image.new(mode,
                    (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs] + outputs

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = (np.array((images[cat][ix]).cpu())*255).astype(np.uint8).squeeze()
            im = Image.fromarray(im, mode)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def postprocess(img, norm=False):
    # [-1, 1] => [0, 255]
    # if img.shape[2] < 160:
    #     img = F.interpolate(img, (160, 160))
#     if img.shape[2] < 128:
#         img = F.interpolate(img, (128, 128))
    if norm:  # to [-1~1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-7)  # [0~1]
        img = img * 2 - 1  # [-1~1]
    # img = (img + 1) / 2 * 255.0
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    # return img.int()
    return img

def show_pose_img(pose):
    pose = pose.squeeze(0)
    pose *= 256
    pose = pose.cpu().numpy().astype(np.uint8)
    pose_img = np.ones((256, 256, 3))*255
    pose_img = pose_img.astype(np.uint8)
#     print(pose.shape)
    for i in range(pose.shape[0]):
        pose_img[pose[i,1]-3:pose[i,1]+3,
                 pose[i, 0]-3:pose[i,0]+3, :] = 0
    
    pose_img = torch.tensor(pose_img)
    return pose_img.int()

def imsave(img, path):
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def compute_color(u, v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
    return img


def convert_offsets_to_colors(offsets, hide_valid=True):
    # offsets: [B,H,W,2]
    [_, H, W, D] = offsets.shape
    assert D == 2
    colors = []
    mcw = make_color_wheel()
    for i in range(offsets.shape[0]):
        offset = offsets[i]  # [H,W,2]
        valid_area = (np.sum(np.abs(offset), axis=2, keepdims=True) == 0).astype(int)  # [H,W,1]
        h_add = np.tile(np.arange(0, H).reshape((H, 1, 1)), (1, W, 1))
        w_add = np.tile(np.arange(0, W).reshape((1, W, 1)), (H, 1, 1))
        offset = offset + np.concatenate([h_add, w_add], axis=-1)
        color = np.zeros((H, W, 3), dtype=np.uint8)
        for h in range(H):
            for w in range(W):
                x = offset[h, w, 1]
                color[h, w] = mcw[int(x / W * 51)]
        color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
        for h in range(H):
            for w in range(W):
                y = offset[h, w, 0]
                color[h, w, 1] = int(color[h, w, 1] * ((1 - (y / H)) * 0.75 + 0.25))
        color = cv2.cvtColor(color, cv2.COLOR_HSV2RGB)
        if hide_valid:
            color = color * (1 - valid_area) + np.ones((H, W, 3), dtype=np.uint8) * 255 * valid_area
        colors.append(color[None])
    colors = np.concatenate(colors, axis=0)
    return colors


def flow_to_image(flow):
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def torch_init_model(model, init_checkpoint, key):
    state_dict = torch.load(init_checkpoint, map_location='cpu')[key]
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='')
    
    print("missing keys:{}".format(missing_keys))
    print('unexpected keys:{}'.format(unexpected_keys))
    print('error msgs:{}'.format(error_msgs))


def get_combined_mask(mask, min_size=16):
    final_mask = None
    max_size = mask.shape[2]
    size = min_size
    while size < max_size:
        m = F.max_pool2d(mask, int(max_size / size), int(max_size / size))
        if final_mask is None:
            final_mask = m
        else:
            final_mask = F.interpolate(final_mask, (size, size), mode='nearest')
            final_mask = torch.cat([final_mask, m], dim=1)
            final_mask = torch.mean(final_mask, dim=1, keepdim=True)
        size *= 2

    final_mask = F.interpolate(final_mask, (max_size, max_size), mode='bilinear')
    return final_mask

