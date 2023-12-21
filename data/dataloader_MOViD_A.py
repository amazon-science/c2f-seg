import os
import cv2
import json
import pickle
import random
import numpy as np
import cvbase as cvb
from PIL import Image
from skimage import transform
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

def mask_find_bboxs(mask):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8) # connectivity参数的默认值为8
    stats = stats[stats[:,4].argsort()]
    return stats[:-1]

class Movid_A_VQ_Dataset(object):
    def __init__(self, config, mode):
        self.mode = mode
        self.config = config
        self.device = "cpu"
        root_path = config.root_path

        self.image_list = np.genfromtxt(os.path.join(root_path,"{}_frame_list.txt".format(mode)), dtype=np.str, encoding='utf-8')
        self.data_dir = config.root_path
        self.dtype = torch.float32
        self.enlarge_coef = 1.2
        self.patch_h = 256
        self.patch_w = 256
        self.Image_H = 256
        self.Image_W = 256

    def __len__(self):
        return len(self.image_list)
            
    def __getitem__(self, idx):
        video_id, obj_id, frame_id = self.image_list[idx].split("_")
        mask_value = int(obj_id)+1
        video_path = os.path.join(self.data_dir, self.mode, str(video_id))
        full_mask_path = os.path.join(video_path, "segmentation_{}_{}.png".format(obj_id, frame_id.zfill(5)))
        full_mask = (np.array(Image.open(full_mask_path))==mask_value)

        metadata = self.read_json(os.path.join(video_path, 'metadata.json'))

        try:
            f_idx = metadata["instances"][int(obj_id)]['bbox_frames'].index(int(frame_id))
            xmin, ymin, xmax, ymax = metadata["instances"][int(obj_id)]["bboxes"][f_idx]
            vx_min, vy_min, vx_max, vy_max = int(self.Image_H*xmin), int(self.Image_W*ymin), int(self.Image_H*xmax), int(self.Image_W*ymax)
            # print(vx_min, vy_min, vx_max, vy_max)
        except:
            bboxs = mask_find_bboxs(full_mask.astype(np.uint8))
            if bboxs.size==0:
                vx_min, vy_min, vx_max, vy_max = 0, 0, 256, 256
            else:
                b = bboxs[-1][:4]
                vx_min, vy_min, vx_max, vy_max = b[1], b[0], b[1]+b[3], b[0]+b[2]
        # enlarge the bbox
        x_center = (vx_min + vx_max) // 2
        y_center = (vy_min + vy_max) // 2
        x_len = int((vx_max - vx_min) * self.enlarge_coef)
        y_len = int((vy_max - vy_min) * self.enlarge_coef)
        vx_min = max(0, x_center - x_len // 2)
        vx_max = min(self.Image_H, x_center + x_len // 2)
        vy_min = max(0, y_center - y_len // 2)
        vy_max = min(self.Image_W, y_center + y_len // 2)
        mask_crop = full_mask[vx_min:vx_max+1, vy_min:vy_max+1]
        
        h, w = mask_crop.shape[:2]
        m = transform.rescale(mask_crop, (self.patch_h/h, self.patch_w/w))
        cur_h, cur_w = m.shape[:2]
        to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
        m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]
        mask_crop = m[..., np.newaxis]
        mask_crop = torch.from_numpy(mask_crop).permute(2, 0, 1).to(self.dtype)
        
        # load iamge for vq
        # img_crop = rgb_img[vx_min:vx_max+1, vy_min:vy_max+1]
        # img = transform.rescale(img_crop, (self.patch_h/h, self.patch_w/w, 1))
        # cur_h, cur_w = img.shape[:2]
        # to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)), (0, 0))
        # img = np.pad(img, to_pad)[:self.patch_h, :self.patch_w, :3]
        # img_crop = img
        # img_crop = torch.from_numpy(img_crop).permute(2, 0, 1).to(self.dtype)
        # # img_crop = img_crop * mask_crop
        
        meta = {
            'mask_crop': mask_crop,
            # 'image_crop': img_crop
        }
        return meta

    def read_json(self,dir_):
        with open(dir_) as f:
            data = json.load(f)
        return data

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

class MOViD_A(object):
    def __init__(self, config, mode):
        super(MOViD_A, self).__init__()
        self.mode = mode
        self.dtype = torch.float32
        self.device = "cpu"
        root_path = config.root_path
        self.data_dir = os.path.join(root_path, mode)
        
        self.instance_list = np.genfromtxt(
            os.path.join(root_path, "{}_instance.txt".format(mode)),
            dtype=np.str,
            encoding='utf-8'
        )

        self.train_seq_len = 24
        self.cur_vid = None
        self.patch_h = config.patch_H
        self.patch_w = config.patch_W
        self.enlarge_coef = config.enlarge_coef

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, idx, specified_V_O_id=None):
        # whether choose a specific instance to load
        if specified_V_O_id is None:
            v_id, obj_id, value = self.instance_list[idx].split("_")
        else:
            v_id, obj_id, value = specified_V_O_id.split("_")
        v_id, obj_id, value = int(v_id), int(obj_id), int(value)
        if v_id != self.cur_vid:
            self.cur_vid = v_id
            self.video_path = os.path.join(self.data_dir, str(v_id))
            metadata = self.read_json(os.path.join(self.video_path, 'metadata.json'))

            self.num_frames = metadata["metadata"]["num_frames"]
            self.height = metadata['metadata']['height']
            self.width = metadata['metadata']['width']
            self.instances = [self.format_instance_information(obj) for obj in metadata["instances"]]

        vis_mask_paths = [os.path.join(self.video_path, "segmentation_full_{}.png".format(str(f).zfill(5))) for f in range(self.num_frames)]
        vis_mask = [np.array(Image.open(frame_path)) for frame_path in vis_mask_paths] #[t,h,w]

        full_mask_paths = [os.path.join(self.video_path, "segmentation_{}_{}.png".format(obj_id, str(f).zfill(5))) for f in range(self.num_frames)]
        full_mask = [np.array(Image.open(frame_path)) for frame_path in full_mask_paths] #[t,h,w]
                
        rgb_img_path = [os.path.join(self.video_path, "rgba_full_{}.png".format(str(f).zfill(5))) for f in range(self.num_frames)]
        rgb_img = [np.array(Image.open(frame_path))[...,:3] for frame_path in rgb_img_path]
        
        counts = []
        obj_position = []

        vm_crop = []
        vm_no_crop = []
        fm_crop = []
        fm_no_crop = []
        loss_mask_weight = []
        img_crop = []
        # for evaluation 
        video_ids = []
        object_ids = []
        frame_ids = []

        timesteps = self.instances[obj_id]['bbox_frames']
        start_t, end_t = 0, 23
        if self.mode != "test" and end_t - start_t > self.train_seq_len - 1:
            start_t = np.random.randint(start_t, end_t-(self.train_seq_len-2))
            end_t = start_t + self.train_seq_len - 1

        for t_step in range(start_t, end_t+1):
            Image_H, Image_W = self.height, self.width
            # some objects will move out the field of view in some frames
            if t_step in timesteps:
                index = self.instances[obj_id]["bbox_frames"].index(t_step)
                xmin, ymin, xmax, ymax = self.instances[obj_id]["bboxes"][index]
                vx_min, vy_min, vx_max, vy_max = int(Image_H*xmin), int(Image_W*ymin), int(Image_H*xmax), int(Image_W*ymax)
                counts.append(1)
            else:
                bboxs = mask_find_bboxs(full_mask[t_step].astype(np.uint8))
            
                if bboxs.size==0:
                    vx_min, vy_min, vx_max, vy_max = 0, 0, 256, 256
                else:
                    b = bboxs[-1][:4]
                    vx_min, vy_min, vx_max, vy_max = b[1], b[0], b[1]+b[3], b[0]+b[2]
                counts.append(0)

            # enlarge the bbox
            x_center = (vx_min + vx_max) // 2
            y_center = (vy_min + vy_max) // 2
            x_len = int((vx_max - vx_min) * self.enlarge_coef)
            y_len = int((vy_max - vy_min) * self.enlarge_coef)
            vx_min = max(0, x_center - x_len // 2)
            vx_max = min(Image_H, x_center + x_len // 2)
            vy_min = max(0, y_center - y_len // 2)
            vy_max = min(Image_W, y_center + y_len // 2)

            obj_position.append([vx_min, vx_max, vy_min, vy_max])

            # get mask
            vm = vis_mask[t_step]
            vm_crop.append(vm[vx_min:vx_max+1, vy_min:vy_max+1]==value)
            vm_no_crop.append(vm==value)

            fm = full_mask[t_step]
            fm_crop.append(fm[vx_min:vx_max+1, vy_min:vy_max+1]==value)
            fm_no_crop.append(fm==value)
            
            # get image
            image = rgb_img[t_step]
            img_crop.append(image[vx_min:vx_max+1, vy_min:vy_max+1])

            # get loss mask
            fore_ground = vm == 0
            obj_ground = vm==value
            loss_mask = np.logical_or(fore_ground, obj_ground)

            loss_mask_weight.append(loss_mask)

            # for evaluation
            video_ids.append(v_id)
            object_ids.append(obj_id)
            frame_ids.append(t_step)

        obj_position = torch.from_numpy(np.array(obj_position)).to(self.dtype).to(self.device)
        
        vm_crop, fm_crop, vm_pad, vm_scale, vm_crop_gt, img_crop = self.crop_and_rescale(vm_crop, fm_crop, img_crop)

        vm_crop = np.stack(vm_crop, axis=0) # Seq_len * h * w
        vm_no_crop = np.stack(vm_no_crop, axis=0) # Seq_len * H * W
        # fm_crop = np.stack(fm_crop, axis=0) # Seq_len * h * w
        fm_crop = np.stack(fm_crop, axis=0) # Seq_len * h * w
        fm_no_crop = np.stack(fm_no_crop, axis=0) # Seq_len * H * W
        img_crop = np.stack(img_crop, axis=0) # Sqe_len * H * W

        vm_crop = torch.from_numpy(np.array(vm_crop)).to(self.dtype).to(self.device)
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
            "vm_no_crop": vm_no_crop,
            "vm_pad": vm_pad,
            "vm_scale": vm_scale,

            "img_crop": img_crop,
            
            "fm_crop": fm_crop,
            "fm_no_crop": fm_no_crop,

            "obj_position": obj_position, 
            "loss_mask": loss_mask_weight, 
            "counts": counts,
            "video_ids": video_ids,
            "object_ids": object_ids,
            "frame_ids": frame_ids,
        }

        return obj_data

    def crop_and_rescale(self, vm_crop, fm_crop=None,img_crop=None):
        h, w = np.array([m.shape for m in vm_crop]).max(axis=0)
        vm_pad = []
        vm_crop_gt = []
        vm_scale = []
        for i, img in enumerate(img_crop):
            img = transform.rescale(img, (self.patch_h/h, self.patch_w/w, 1))
            cur_h, cur_w = img.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)), (0, 0))
            img = np.pad(img, to_pad)[:self.patch_h, :self.patch_w, :3]
            img_crop[i] = img

        for i, m in enumerate(vm_crop):
            m = transform.rescale(m, (self.patch_h/h, self.patch_w/w))
            cur_h, cur_w = m.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
            m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]
            if self.mode=="train":
                vm_crop[i] = self.data_augmentation(m)
            else:
                vm_crop[i] = m
            vm_crop_gt.append(m)
            vm_pad.append(np.array([max(self.patch_h-cur_h, 0), max(self.patch_w-cur_w, 0)]))
            vm_scale.append(np.array([self.patch_h/h, self.patch_w/w]))

        for i, m in enumerate(fm_crop):
            m = transform.rescale(m, (self.patch_h/h, self.patch_w/w))
            cur_h, cur_w = m.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
            m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]
            fm_crop[i] = m

        vm_pad = np.stack(vm_pad)
        vm_scale = np.stack(vm_scale)
        return vm_crop, fm_crop, vm_pad, vm_scale, vm_crop_gt,img_crop
    
    def read_json(self,dir_):
        with open(dir_) as f:
            data = json.load(f)
        return data

    def format_instance_information(self, obj):
        return {
            "bboxes": obj["bboxes"],
            "bbox_frames": obj["bbox_frames"],
        }

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