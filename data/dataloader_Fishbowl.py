import os
import cv2
import glob
import copy
import pickle
import random
import numpy as np
import cvbase as cvb
from PIL import Image
from skimage import transform
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import pycocotools.mask as mask_utils


class FishBowl_VQ_Dataset(torch.utils.data.Dataset):
    def __init__(self, config, mode="train"):
        super(FishBowl_VQ_Dataset, self).__init__()
        self.config = config
        root_path = config.root_path
        self.data = self.load_flist(os.path.join(root_path, "{}_list.txt".format(mode)))
        self.image_root = os.path.join(root_path, "{}_data/{}_frames".format(mode, mode))
        self.mask = pickle.load(open(os.path.join(root_path,"{}_fish.pkl".format(mode)), 'rb'))
        self.dtype = torch.float32
        self.enlarge_coef = 1.5
        self.patch_h = 256
        self.patch_w = 256

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.load_item(index)
        
    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        # load image
        obj_name = self.data[index] # 00043_1_0
        v_id = obj_name.split('_')[0] # 00043
        f_id = obj_name.split('_')[-1].zfill(5) # 00000
        if self.config.mask_type=="FM":
            mask = self.decode2binarymask(self.mask[obj_name]["FM"])[0]
            x_min, x_max, y_min, y_max = self.mask[obj_name]["FM_bx"]
        elif self.config.mask_type=="VM":
            mask = self.decode2binarymask(self.mask[obj_name]["VM"])[0]
            x_min, x_max, y_min, y_max = self.mask[obj_name]["VM_bx"]

        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        x_len = int((x_max - x_min) * self.enlarge_coef)
        y_len = int((y_max - y_min) * self.enlarge_coef)
        x_min = max(0, x_center - x_len // 2)
        x_max = min(320, x_center + x_len // 2)
        y_min = max(0, y_center - y_len // 2)
        y_max = min(480, y_center + y_len // 2)
        mask_crop = mask[x_min:x_max+1, y_min:y_max+1]

        h, w = mask_crop.shape[:2]
        m = transform.rescale(mask_crop, (self.patch_h/h, self.patch_w/w))
        cur_h, cur_w = m.shape[:2]
        to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
        m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]
        mask_crop = m[..., np.newaxis]
        mask_crop = torch.from_numpy(mask_crop).permute(2, 0, 1).to(self.dtype)

        image_path = os.path.join(self.image_root, v_id, f_id+'.png') 
        img = cv2.imread(image_path)[:,:,::-1]
        img_crop = img[x_min:x_max+1, y_min:y_max+1]
        m = transform.rescale(img_crop, (self.patch_h/h, self.patch_w/w, 1))
        cur_h, cur_w = m.shape[:2]
        to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)), (0, 0))
        m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w, :3]
        img_crop = m
        
        img_crop = torch.from_numpy(img_crop).permute(2, 0, 1).to(self.dtype)
        img_crop = img_crop * mask_crop

        meta = {
            'mask_crop': mask_crop,
            'image_crop': img_crop
        }

        return meta

    def to_tensor(self, img, norm=False):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        if norm:
            img_t = F.normalize(img_t, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        return img_t

    def resize(self, img, height, width, center_crop=False):
        imgh, imgw = img.shape[0:2]

        if center_crop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        if imgh > height and imgw > width:
            inter = cv2.INTER_AREA 
        else:
            inter = cv2.INTER_LINEAR 
        img = cv2.resize(img, (height, width), interpolation=inter)

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist
        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort(key=lambda x: x.split('/')[-1])
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                collate_fn=self.collate_fn
            )

            for item in sample_loader:
                yield item

    @staticmethod
    def collate_fn(batch):
        keys = batch[0].keys()
        res = {}
        for k in keys:
            temp_ = []
            for b in batch:
                if b[k] is not None:
                    temp_.append(b[k])
            if len(temp_) > 0:
                res[k] = default_collate(temp_)
            else:
                res[k] = None

        return res

    def decode2binarymask(self, masks):
        mask = mask_utils.decode(masks)
        binary_masks = mask.astype('bool') # (Image_W,Image_H,128)
        binary_masks = binary_masks.transpose(2,0,1) #(128, Image_W, Image_H)
        return binary_masks

class FishBowl(object):
    def __init__(self, config, mode, subtest=None):
        self.datatype = mode
        data_dir = config.root_path

        self.img_path = os.path.join(data_dir, self.datatype+"_data", self.datatype+"_frames")
        self.mode = mode
        self.dtype = torch.float32
        self.test_set = subtest
    
        self.data_summary = pickle.load(open(os.path.join(data_dir, self.datatype+"_data", self.datatype+"_data.pkl"), "rb"))
        self.obj_lists = list(self.data_summary.keys())
        self.device = "cpu"

        self.seq_len = 32 if self.mode == "test" else config.train_seq_len

        self.cur_vid = None
        self.video_frames = None
        self.patch_h = config.patch_H
        self.patch_w = config.patch_W
        self.enlarge_coef = config.enlarge_coef

    def decode2binarymask(self, masks):
        mask = mask_utils.decode(masks)
        binary_masks = mask.astype('bool') # (Image_W,Image_H,128)
        binary_masks = binary_masks.transpose(2,0,1) #(128, Image_W, Image_H)
        return binary_masks

    def __len__(self):
        return len(self.obj_lists)

    def __getitem__(self, idx):
        v_id, obj_id = self.obj_lists[idx].split("_")
        if v_id != self.cur_vid:
            self.cur_vid = v_id
        fm_crop = []
        fm_no_crop = []
        vm_crop = []
        vm_no_crop = []
        img_crop = []
        
        obj_position = []

        counts = []
        loss_mask_weight = []

        # for evaluation 
        video_ids = []
        object_ids = []
        frame_ids = []

        obj_dict = self.data_summary[self.obj_lists[idx]]
        timesteps = list(obj_dict.keys())
        assert np.all(np.diff(sorted(timesteps))==1)
        start_t, end_t = min(timesteps), max(timesteps)
        # print(start_t, end_t)
        if self.mode != "test" and end_t - start_t > self.seq_len - 1:
            start_t = np.random.randint(start_t, end_t-(self.seq_len-2))
            end_t = start_t + self.seq_len - 1

        if self.mode == "test":
            if start_t + self.seq_len-1<=end_t:
                end_t = start_t + self.seq_len-1

        for t_step in range(start_t, end_t):
            image_path = os.path.join(self.img_path, v_id, str(t_step).zfill(5)+'.png')
            img = cv2.imread(image_path)[:,:,::-1]
            # get visible mask and full mask
            vm = self.decode2binarymask(obj_dict[t_step]["VM"])[0]
            fm = self.decode2binarymask(obj_dict[t_step]["FM"])[0] # 320, 480
            vx_min, vx_max, vy_min, vy_max = obj_dict[t_step]["VM_bx"]
            x_center = (vx_min + vx_max) // 2
            y_center = (vy_min + vy_max) // 2
            x_len = int((vx_max - vx_min) * self.enlarge_coef)
            y_len = int((vy_max - vy_min) * self.enlarge_coef)
            vx_min = max(0, x_center - x_len // 2)
            vx_max = min(320, x_center + x_len // 2)
            vy_min = max(0, y_center - y_len // 2)
            vy_max = min(480, y_center + y_len // 2)

            obj_position.append([vx_min, vx_max, vy_min, vy_max])
            vm_crop.append(vm[vx_min:vx_max+1, vy_min:vy_max+1])
            fm_crop.append(fm[vx_min:vx_max+1, vy_min:vy_max+1])
            img_crop.append(img[vx_min:vx_max+1, vy_min:vy_max+1])

            vm_no_crop.append(vm)
            fm_no_crop.append(fm)
            # get loss mask
            loss_mask_weight.append(self.decode2binarymask(obj_dict[t_step]["loss_mask_weight"])[0])

            # for evaluation
            video_ids.append(int(v_id))
            object_ids.append(int(obj_id))
            frame_ids.append(t_step)
            counts.append(1)
        
        if True:
            num_pad = self.seq_len - (end_t - start_t)
            for _  in range(num_pad):
                obj_position.append(copy.deepcopy(obj_position[-1]))

                fm_crop.append(copy.deepcopy(fm_crop[-1]))
                fm_no_crop.append(copy.deepcopy(fm_no_crop[-1]))
                vm_crop.append(copy.deepcopy(vm_crop[-1]))
                vm_no_crop.append(copy.deepcopy(vm_no_crop[-1]))
                img_crop.append(copy.deepcopy(img_crop[-1]))

                loss_mask_weight.append(copy.deepcopy(loss_mask_weight[-1]))
                
                video_ids.append(video_ids[-1])
                object_ids.append(object_ids[-1])
                frame_ids.append(frame_ids[-1] + 1)
                counts.append(0)
        
        vm_crop, vm_crop_gt, fm_crop, img_crop, vm_pad, vm_scale = self.crop_and_rescale(vm_crop, fm_crop, img_crop)

        vm_crop = np.stack(vm_crop, axis=0) # Seq_len * h * w
        vm_crop_gt = np.stack(vm_crop_gt, axis=0) # Seq_len * h * w
        vm_no_crop = np.stack(vm_no_crop, axis=0) # Seq_len * H * W
        fm_crop = np.stack(fm_crop, axis=0) # Seq_len * h * w
        fm_no_crop = np.stack(fm_no_crop, axis=0) # Seq_len * H * W

        vm_crop = torch.from_numpy(np.array(vm_crop)).to(self.dtype).to(self.device)
        vm_crop_gt = torch.from_numpy(np.array(vm_crop_gt)).to(self.dtype).to(self.device)
        vm_no_crop = torch.from_numpy(np.array(vm_no_crop)).to(self.dtype).to(self.device)
        fm_crop = torch.from_numpy(np.array(fm_crop)).to(self.dtype).to(self.device)
        fm_no_crop = torch.from_numpy(np.array(fm_no_crop)).to(self.dtype).to(self.device)
        img_crop = torch.from_numpy(np.array(img_crop)).to(self.dtype).to(self.device)

        vm_pad = torch.from_numpy(np.array(vm_pad)).to(self.dtype).to(self.device)
        vm_scale = torch.from_numpy(np.array(vm_scale)).to(self.dtype).to(self.device)

        video_ids = torch.from_numpy(np.array(video_ids)).to(self.dtype).to(self.device)
        object_ids = torch.from_numpy(np.array(object_ids)).to(self.dtype).to(self.device)
        frame_ids = torch.from_numpy(np.array(frame_ids)).to(self.dtype).to(self.device)
        counts = torch.from_numpy(np.array(counts)).to(self.dtype).to(self.device)
        loss_mask_weight = torch.from_numpy(np.array(loss_mask_weight)).to(self.dtype).to(self.device) 
        obj_position = torch.from_numpy(np.array(obj_position)).to(self.dtype).to(self.device)

        obj_data = {
            "vm_crop": vm_crop,
            "vm_crop_gt": vm_crop_gt,
            "vm_no_crop": vm_no_crop,
            "fm_crop": fm_crop,
            "fm_no_crop": fm_no_crop,
            "img_crop": img_crop,
            "vm_pad": vm_pad,
            "vm_scale": vm_scale,
            "video_ids": video_ids,
            "object_ids": object_ids,
            "frame_ids": frame_ids,
            "counts": counts,
            "loss_mask": loss_mask_weight, 
            "obj_position": obj_position,
        }

        return obj_data

    def crop_and_rescale(self, vm_crop, fm_crop_vm=None, img_crop=None):
        h, w = np.array([m.shape for m in vm_crop]).max(axis=0)
        vm_pad = []
        vm_scale = []
        vm_crop_gt = []

        for i, m in enumerate(vm_crop):
            m = transform.rescale(m, (self.patch_h/h, self.patch_w/w))
            cur_h, cur_w = m.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
            vm_pad.append(np.array([max(self.patch_h-cur_h, 0), max(self.patch_w-cur_w, 0)]))
            vm_scale.append(np.array([self.patch_h/h, self.patch_w/w]))
            m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]
            if self.mode=="train":
                vm_crop[i] = self.data_augmentation(m)
                vm_crop_gt.append(m)
            else:
                vm_crop[i] = m
                vm_crop_gt.append(m)

        for i, m in enumerate(fm_crop_vm):
            m = transform.rescale(m, (self.patch_h/h, self.patch_w/w))
            cur_h, cur_w = m.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
            m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]
            fm_crop_vm[i] = m

        for i, img_ in enumerate(img_crop):
            img_ = transform.rescale(img_, (self.patch_h/h, self.patch_w/w, 1))
            cur_h, cur_w = img_.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)), (0, 0))
            img_ = np.pad(img_, to_pad)[:self.patch_h, :self.patch_w, :3]
            img_crop[i] = img_

        vm_pad = np.stack(vm_pad)
        vm_scale = np.stack(vm_scale)
        return vm_crop, vm_crop_gt, fm_crop_vm, img_crop, vm_pad, vm_scale
    
    def getImg(self, v_id):
        imgs = []
        imgs_list = os.listdir(os.path.join(self.img_path, v_id))
        imgs_list.sort()
        for sub_path in imgs_list:
            img_path = os.path.join(self.img_path, v_id, sub_path)
            img_tmp = plt.imread(img_path)
            imgs.append(img_tmp)
        assert len(imgs) == 128
        return imgs

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                collate_fn=self.collate_fn
            )
            for item in sample_loader:
                yield item
    
    @staticmethod
    def collate_fn(batch):
        keys = batch[0].keys()
        res = {}
        for k in keys:
            temp_ = []
            for b in batch:
                if b[k] is not None:
                    temp_.append(b[k])
            if len(temp_) > 0:
                res[k] = default_collate(temp_)
            else:
                res[k] = None
        return res
    
    def data_augmentation(self, mask):
        mask = mask.astype(np.float)
        rdv = random.random()
        n_repeat = random.randint(1, 4)
        if rdv <= 0.1:
            mask = cv2.GaussianBlur(mask, (35,35), 11)
        elif rdv > 0.1 and rdv < 0.6:
            rdv_1 = random.random()
            rdv_2 = random.random()
            for i in range(n_repeat):
                w = random.randint(5, 13)
                h = random.randint(5, 13)
                kernel = np.ones((w, h), dtype=np.uint8)
                if rdv_1 <= 0.5:
                    mask = cv2.dilate(mask, kernel, 1)
                elif rdv_1 > 0.5 and rdv_1 <= 1.0:
                    mask = cv2.erode(mask, kernel, 1)
                if rdv_2 <= 0.1:
                    mask = cv2.GaussianBlur(mask, (35,35), 11)
        else:
            mask = mask
        return (mask>0.5)