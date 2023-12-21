import os
import cv2
import pickle
import random
import numpy as np
import cvbase as cvb
from PIL import Image
from skimage import transform
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import pycocotools.mask as mask_utils


class COCOA_VQ_dataset(torch.utils.data.Dataset):
    def __init__(self, config, mode):
        super(COCOA_VQ_dataset, self).__init__()
        self.config = config
        self.mode = mode
        root_path = config.root_path

        if mode=="train":
            train_label = cvb.load(os.path.join(root_path, "COCO_amodal_train2014_with_classes.json"))
            self.anns_dict = train_label["annotations"]
            flist = os.path.join(root_path, "vq_train_list.txt")
        elif mode=="test":
            val_label = cvb.load(os.path.join(root_path, "COCO_amodal_val2014_with_classes.json"))
            self.anns_dict = val_label["annotations"]
            flist = os.path.join(root_path, "vq_val_list.txt")
            
        self.image_list = np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
        self.data_list = list(self.anns_dict)
        self.dtype = torch.float32
        self.enlarge_coef = 2
        self.patch_h = 256
        self.patch_w = 256
        self.device = "cpu"

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        return self.load_item(index)
        
    def load_item(self, index):
        if self.mode=="train":
            self.enlarge_coef = random.uniform(1.5, 3)
        anno_idx, img_path = self.image_list[index].split(",")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = img[...,::-1]
        height, width, _ = img.shape
        
        ann = self.anns_dict[int(anno_idx)]

        full_mask = ann["segmentation"]
        fm_no_crop = mask_utils.decode(full_mask)

        y_min, x_min, w, h = ann["bbox"]
        y_max, x_max = y_min + w, x_min + h
        y_min, x_min, y_max, x_max = int(y_min), int(x_min), int(y_max), int(x_max) 

        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        x_len = int((x_max - x_min) * self.enlarge_coef)
        y_len = int((y_max - y_min) * self.enlarge_coef)
        x_min = max(0, x_center - x_len // 2)
        x_max = min(height, x_center + x_len // 2)
        y_min = max(0, y_center - y_len // 2)
        y_max = min(width, y_center + y_len // 2)
        
        fm_crop = fm_no_crop[x_min:x_max+1, y_min:y_max+1].astype(bool)
        h, w = fm_crop.shape[:2]
        m = transform.rescale(fm_crop, (self.patch_h/h, self.patch_w/w))
        cur_h, cur_w = m.shape[:2]
        to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
        m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]    
        fm_crop = m[np.newaxis, ...]

        fm_crop = torch.from_numpy(fm_crop).to(self.dtype).to(self.device)

        meta = {
            "mask_crop": fm_crop,
        }

        return meta

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

    def make_json_dict(self, imgs, anns):
        imgs_dict = {}
        anns_dict = {}
        for ann in anns:
            image_id = ann["image_id"]
            if not image_id in anns_dict:
                anns_dict[image_id] = []
                anns_dict[image_id].append(ann)
            else:
                anns_dict[image_id].append(ann)
        
        for img in imgs:
            image_id = img['id']
            imgs_dict[image_id] = img['file_name']

        return imgs_dict, anns_dict

    def polys_to_mask(self, polygons, height, width):
        rles = mask_utils.frPyObjects(polygons, height, width)
        rle = mask_utils.merge(rles)
        mask = mask_utils.decode(rle)
        return mask

class COCOA_Fusion_dataset(torch.utils.data.Dataset):
    def __init__(self, config, mode):
        super(COCOA_Fusion_dataset, self).__init__()
        self.config = config
        self.mode = mode
        self.root_path = config.root_path
        
        # Load Fusion dataset 
        self.data_info = pickle.load(open(os.path.join(self.root_path, "fusion_{}.pkl".format(self.mode)), "rb"))
        self.label_info = np.genfromtxt(os.path.join(self.root_path, "c2f_seg_{}_list.txt".format(self.mode)), dtype=np.str, encoding='utf-8')
        
        if mode=="train":
            train_label = cvb.load(os.path.join(self.root_path, "COCO_amodal_train2014_with_classes.json"))
            self.anns_dict = train_label["annotations"]
            self.img_root_path = os.path.join(self.root_path, "train2014")
        elif mode=="test":
            val_label = cvb.load(os.path.join(self.root_path, "COCO_amodal_val2014_with_classes.json"))
            self.anns_dict = val_label["annotations"]
            self.img_root_path = os.path.join(self.root_path, "val2014")
            
        self.dtype = torch.float32
        self.enlarge_coef = 2
        self.patch_h = 256
        self.patch_w = 256
        self.device = "cpu"

        
    def __len__(self):
        return self.label_info.shape[0]

    def __getitem__(self, index):
        return self.load_item(index)
        
    def load_item(self, index):
        # predicted vm
        if len(self.label_info[index].split(","))==3:
            dataset_name, image_id, anno_id = self.label_info[index].split(",")
            image_id, anno_id = int(image_id), int(anno_id)
            if self.mode=="train":
                img_path = os.path.join(self.img_root_path, "COCO_{}2014_{}.jpg".format(self.mode, str(image_id).zfill(12)))
            elif self.mode=="test":
                img_path = os.path.join(self.img_root_path, "COCO_val2014_{}.jpg".format(str(image_id).zfill(12)))
            img = np.array(Image.open(img_path))
            if len(img.shape)==2:
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            instances = self.data_info["{}_{}".format(dataset_name, image_id)][anno_id]
            segmentation = instances["pred_visible_mask"]
            height, weight = segmentation["size"]
            # occlude_rate = instances["occlude_rate"]
            vm_no_crop = mask_utils.decode([segmentation]).astype(bool)
            fm_no_crop = mask_utils.decode([instances["gt_full_mask"]]).astype(bool)
            vm_no_crop_gt = mask_utils.decode([instances["gt_visible_mask"]]).astype(bool)

            bbox = instances["pred_visible_mask_bbox"]
            y_min, x_min, w, h = bbox
            y_max, x_max = y_min + w, x_min + h
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            x_len = int((x_max - x_min) * self.enlarge_coef)
            y_len = int((y_max - y_min) * self.enlarge_coef)
            x_min = max(0, x_center - x_len // 2)
            x_max = min(height, x_center + x_len // 2)
            y_min = max(0, y_center - y_len // 2)
            y_max = min(weight, y_center + y_len // 2)
            x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
            # import pdb;pdb.set_trace()
            vm_crop = vm_no_crop[x_min:x_max+1, y_min:y_max+1, 0].astype(bool)
            fm_crop = fm_no_crop[x_min:x_max+1, y_min:y_max+1, 0].astype(bool)
            img_crop = img[x_min:x_max+1, y_min:y_max+1]
            vm_crop_gt = vm_no_crop_gt[x_min:x_max+1, y_min:y_max+1, 0].astype(bool)

            h, w = vm_crop.shape[:2]
            m = transform.rescale(vm_crop, (self.patch_h/h, self.patch_w/w))
            cur_h, cur_w = m.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
            m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]
            vm_crop = m[np.newaxis, ...]

            img_ = transform.rescale(img_crop, (self.patch_h/h, self.patch_w/w, 1))
            cur_h, cur_w = img_.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)), (0, 0))
            img_ = np.pad(img_, to_pad)[:self.patch_h, :self.patch_w, :3]
            img_crop = img_

            h, w = vm_crop_gt.shape[:2]
            m = transform.rescale(vm_crop_gt, (self.patch_h/h, self.patch_w/w))
            cur_h, cur_w = m.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
            m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]
            vm_crop_gt = m[np.newaxis, ...]

            # data augmentation
            vm_crop_aug = self.data_augmentation(vm_crop[0])[np.newaxis, ...]

            m = transform.rescale(fm_crop, (self.patch_h/h, self.patch_w/w))
            cur_h, cur_w = m.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
            m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]    
            fm_crop = m[np.newaxis, ...]
            # if self.mode=="test":
            #     loss_mask = mask_utils.decode([instances["loss_mask"]]).astype(bool)[...,0]
            # else:
            loss_mask = fm_no_crop.astype(int)-vm_no_crop_gt.astype(int)
            loss_mask[loss_mask==255]=0
            loss_mask = 1-loss_mask.astype(bool)

            vm_no_crop = vm_no_crop[np.newaxis, ...]
            fm_no_crop = fm_no_crop[np.newaxis, ...]

            obj_position = np.array([x_min, x_max, y_min, y_max])
            vm_pad = np.array([max(self.patch_h-cur_h, 0), max(self.patch_w-cur_w, 0)])
            vm_scale = np.array([self.patch_h/h, self.patch_w/w])
            counts = np.array([1])
            counts = torch.from_numpy(counts).to(self.dtype).to(self.device)

            obj_position = torch.from_numpy(obj_position).to(self.dtype).to(self.device)
            vm_pad = torch.from_numpy(vm_pad).to(self.dtype).to(self.device)
            vm_scale = torch.from_numpy(vm_scale).to(self.dtype).to(self.device)

            fm_crop = torch.from_numpy(fm_crop).to(self.dtype).to(self.device)
            fm_no_crop = torch.from_numpy(np.array(fm_no_crop)).to(self.dtype).to(self.device)
            vm_crop = torch.from_numpy(vm_crop).to(self.dtype).to(self.device)
            vm_crop_gt = torch.from_numpy(vm_crop_gt).to(self.dtype).to(self.device)
            vm_crop_aug = torch.from_numpy(vm_crop_aug).to(self.dtype).to(self.device)
            vm_no_crop = torch.from_numpy(np.array(vm_no_crop)).to(self.dtype).to(self.device)

            img_crop = torch.from_numpy(np.array(img_crop)).to(self.dtype).to(self.device)
            img = torch.from_numpy(np.array(img)).to(self.dtype).to(self.device)
            loss_mask = torch.from_numpy(np.array(loss_mask)).to(self.dtype).to(self.device)
        
            image_id = torch.from_numpy(np.array(image_id)).to(self.dtype).to(self.device)
            anno_id = torch.from_numpy(np.array(anno_id)).to(self.dtype).to(self.device)
            # occlude_rate = torch.from_numpy(np.array(occlude_rate)).to(self.dtype).to(self.device)
            
            if self.mode=="train":
                meta = {
                    # "vm_no_crop": vm_no_crop,
                    # "vm_crop": vm_crop,
                    "vm_crop": vm_crop_aug,
                    "vm_crop_gt": vm_crop_gt,
                    # "vm_crop_gt": vm_crop_gt,
                    # "fm_no_crop": fm_no_crop,
                    "fm_crop": fm_crop,
                    "img_crop": img_crop,
                    # "loss_mask": loss_mask,
                    "obj_position": obj_position,
                    "vm_pad": vm_pad,
                    "vm_scale": vm_scale,
                    "counts":counts,
                    "img_id": image_id,
                    "anno_id": anno_id,
                    # for vq
                    # "mask_crop": fm_crop
                    # "img_no_crop": img,
                }
            elif self.mode=="test":
                meta = {
                    "vm_no_crop": vm_no_crop,
                    "vm_crop": vm_crop,
                    "img_crop": img_crop,
                    "vm_crop_gt": vm_crop_gt,
                    "fm_no_crop": fm_no_crop,
                    "fm_crop": fm_crop,
                    "loss_mask": loss_mask,
                    "obj_position": obj_position,
                    "vm_pad": vm_pad,
                    "vm_scale": vm_scale,
                    "counts":counts,
                    "img_id": image_id,
                    "anno_id": anno_id,
                    # "occlude_rate":occlude_rate
                    # for vq
                    # "mask_crop": fm_crop
                    # "img_no_crop": img,
                }
            return meta
        # gt vm
        elif len(self.label_info[index].split(","))==2:
            anno_id, img_path = self.label_info[index].split(",")
            anno_id = int(anno_id)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            height, width, _ = img.shape

            ann = self.anns_dict[anno_id]
            img_id = ann["image_id"]
            # category_id = ann["category_id"]

            full_mask = ann["segmentation"]
            fm_no_crop = mask_utils.decode(full_mask)[...,np.newaxis]

            visible_mask = ann["visible_mask"]
            vm_no_crop = mask_utils.decode(visible_mask)[...,np.newaxis]

            if np.sum(vm_no_crop)==0:
                counts = np.array([0])
            else:
                counts = np.array([1])
            y_min, x_min, w, h = ann["bbox"]
            y_max, x_max = y_min + w, x_min + h
            y_min, x_min, y_max, x_max = int(y_min), int(x_min), int(y_max), int(x_max) 

            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            x_len = int((x_max - x_min) * self.enlarge_coef)
            y_len = int((y_max - y_min) * self.enlarge_coef)
            x_min = max(0, x_center - x_len // 2)
            x_max = min(height, x_center + x_len // 2)
            y_min = max(0, y_center - y_len // 2)
            y_max = min(width, y_center + y_len // 2)
            
            fm_crop = fm_no_crop[x_min:x_max+1, y_min:y_max+1, 0].astype(bool)
            vm_crop = vm_no_crop[x_min:x_max+1, y_min:y_max+1, 0].astype(bool)
            img_crop = img[x_min:x_max+1, y_min:y_max+1]

            h, w = vm_crop.shape[:2]
            m = transform.rescale(vm_crop, (self.patch_h/h, self.patch_w/w))
            cur_h, cur_w = m.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
            m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]
            vm_crop = m[np.newaxis, ...]

            img_ = transform.rescale(img_crop, (self.patch_h/h, self.patch_w/w, 1))
            cur_h, cur_w = img_.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)), (0, 0))
            img_ = np.pad(img_, to_pad)[:self.patch_h, :self.patch_w, :3]
            img_crop = img_

            m = transform.rescale(fm_crop, (self.patch_h/h, self.patch_w/w))
            cur_h, cur_w = m.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
            m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]    
            fm_crop = m[np.newaxis, ...]

            obj_position = np.array([x_min, x_max, y_min, y_max])
            vm_pad = np.array([max(self.patch_h-cur_h, 0), max(self.patch_w-cur_w, 0)])
            vm_scale = np.array([self.patch_h/h, self.patch_w/w])

            # full_pad = ((0, max(375-height, 0)), (0, max(1242-width, 0)))
            # vm_no_crop = np.pad(vm_no_crop, full_pad)[:375, :1242]
            # fm_no_crop = np.pad(fm_no_crop, full_pad)[:375, :1242]
            vm_no_crop = vm_no_crop[np.newaxis, ...]
            fm_no_crop = fm_no_crop[np.newaxis, ...]

            loss_mask = fm_no_crop-vm_no_crop
            loss_mask[loss_mask==255]=0
            loss_mask = 1-loss_mask.astype(bool)
            # data augmentation
            vm_crop_aug = self.data_augmentation(vm_crop[0])[np.newaxis, ...]
            
            counts = torch.from_numpy(counts).to(self.dtype).to(self.device)

            obj_position = torch.from_numpy(obj_position).to(self.dtype).to(self.device)
            vm_pad = torch.from_numpy(vm_pad).to(self.dtype).to(self.device)
            vm_scale = torch.from_numpy(vm_scale).to(self.dtype).to(self.device)

            fm_crop = torch.from_numpy(fm_crop).to(self.dtype).to(self.device)
            fm_no_crop = torch.from_numpy(np.array(fm_no_crop)).to(self.dtype).to(self.device)
            vm_crop = torch.from_numpy(vm_crop).to(self.dtype).to(self.device)
            vm_crop_aug = torch.from_numpy(vm_crop_aug).to(self.dtype).to(self.device)
            img_crop = torch.from_numpy(img_crop).to(self.dtype).to(self.device)
            img = torch.from_numpy(img).to(self.dtype).to(self.device)
            vm_no_crop = torch.from_numpy(np.array(vm_no_crop)).to(self.dtype).to(self.device)
            
            loss_mask = torch.from_numpy(np.array(loss_mask)).to(self.dtype).to(self.device)
        
            img_id = torch.from_numpy(np.array(img_id)).to(self.dtype).to(self.device)
            anno_id = torch.from_numpy(np.array(anno_id)).to(self.dtype).to(self.device)
            # category_id = torch.from_numpy(np.array(category_id)).to(self.dtype).to(self.device)
            if self.mode=="train":
                meta = {
                    # "vm_no_crop": vm_no_crop,
                    "vm_crop": vm_crop_aug,
                    "vm_crop_gt": vm_crop,
                    # "fm_no_crop": fm_no_crop,
                    "fm_crop": fm_crop,
                    "img_crop": img_crop,
                    # "loss_mask": loss_mask,
                    "obj_position": obj_position,
                    "vm_pad": vm_pad,
                    "vm_scale": vm_scale,
                    "counts": counts,
                    "img_id": img_id,
                    "anno_id": anno_id,
                    # "category_id": category_id,
                    # for vq
                    # "mask_crop": fm_crop
                    # "img_no_crop": img
                }
            elif self.mode=="test":
                meta = {
                    "vm_no_crop": vm_no_crop,
                    "vm_no_crop_gt": vm_no_crop,
                    "vm_crop": vm_crop,
                    "vm_crop_gt": vm_crop,
                    "fm_no_crop": fm_no_crop,
                    "fm_crop": fm_crop,
                    "img_crop": img_crop,
                    "loss_mask": loss_mask,
                    "obj_position": obj_position,
                    "vm_pad": vm_pad,
                    "vm_scale": vm_scale,
                    "counts":counts,
                    "img_id": img_id,
                    "anno_id": anno_id,
                    # "category_id": category_id,
                    # for vq
                    # "mask_crop": fm_crop
                    "img_no_crop": img,
                }
            return meta
        
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

    def polys_to_mask(self, polygons, height, width):
        rles = mask_utils.frPyObjects(polygons, height, width)
        rle = mask_utils.merge(rles)
        mask = mask_utils.decode(rle)
        return mask

    # def data_augmentation(self, mask):
    #     return mask
    
    def data_augmentation(self, mask):
        mask = mask.astype(np.float)
        rdv = random.random()
        n_repeat = random.randint(1, 4)
        if rdv <= 0.2:
            mask = cv2.GaussianBlur(mask, (35,35), 11)
        elif rdv > 0.2 and rdv <0.9:
            rdv_1 = random.random()
            rdv_2 = random.random()
            for i in range(n_repeat):
                w = random.randint(5, 13)
                h = random.randint(5, 13)
                kernel = np.ones((w, h), dtype=np.uint8)
                if rdv_1 <= 0.6:
                    mask = cv2.dilate(mask, kernel, 1)
                elif rdv_1 > 0.6 and rdv_1 <= 1.0:
                    mask = cv2.erode(mask, kernel, 1)
                if rdv_2 <= 0.2:
                    mask = cv2.GaussianBlur(mask, (35,35), 11)
        else:
            mask = mask
        return (mask>0.5)
    
    def make_json_dict(self, imgs, anns):
        imgs_dict = {}
        anns_dict = {}
        for ann in anns:
            image_id = ann["image_id"]
            if not image_id in anns_dict:
                anns_dict[image_id] = []
                anns_dict[image_id].append(ann)
            else:
                anns_dict[image_id].append(ann)
        
        for img in imgs:
            image_id = img['id']
            imgs_dict[image_id] = img['file_name']

        return imgs_dict, anns_dict

class COCOA_VRSP(torch.utils.data.Dataset):
    def __init__(self, config, mode):
        super(COCOA_VRSP, self).__init__()
        self.config = config
        self.mode = mode
        self.data_info = pickle.load(open(os.path.join(self.root_path, "fusion_{}.pkl".format(self.mode)), "rb"))
        self.label_info = np.genfromtxt(os.path.join(self.root_path, "c2f_seg_{}_list.txt".format(self.mode)), dtype=np.str, encoding='utf-8')
        
        if self.mode=="train":
            self.img_root_path = os.path.join(self.root_path, "train2014")
        elif self.mode=="test":
            self.img_root_path = os.path.join(self.root_path, "val2014")

        self.dtype = torch.float32
        self.enlarge_coef = 2
        self.patch_h = 256
        self.patch_w = 256
        self.device = "cpu"

        
    def __len__(self):
        return self.label_info.shape[0]

    def __getitem__(self, index):
        return self.load_item(index)
    
    def generate_heatmap(self, mask, kernel, sigma):
        heatmap = cv2.GaussianBlur(mask, kernel, sigma)
        am = np.amax(heatmap)
        heatmap /= am / 1
        return heatmap
    
    def load_item(self, index):
        image_id, anno_id = self.label_info[index].split("_")
        image_id, anno_id = int(image_id), int(anno_id)
        if self.mode=="train":
            img_path = os.path.join(self.img_root_path, "COCO_{}2014_{}.jpg".format(self.mode, str(image_id).zfill(12)))
        elif self.mode=="test":
            img_path = os.path.join(self.img_root_path, "COCO_val2014_{}.jpg".format(str(image_id).zfill(12)))
        img = np.array(Image.open(img_path))
        if len(img.shape)==2:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        instances = self.data_info[image_id][anno_id]
        segmentation = instances["pred_visible_mask"]
        height, weight = segmentation["size"]
        occlude_rate = instances["occlude_rate"]
        vm_no_crop = mask_utils.decode([segmentation]).astype(bool)
        fm_no_crop = mask_utils.decode([instances["gt_full_mask"]]).astype(bool)
        vm_no_crop_gt = mask_utils.decode([instances["gt_visible_mask"]]).astype(bool)

        bbox = instances["pred_visible_mask_bbox"]
        y_min, x_min, w, h = bbox
        y_max, x_max = y_min + w, x_min + h
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        x_len = int((x_max - x_min) * self.enlarge_coef)
        y_len = int((y_max - y_min) * self.enlarge_coef)
        x_min = max(0, x_center - x_len // 2)
        x_max = min(height, x_center + x_len // 2)
        y_min = max(0, y_center - y_len // 2)
        y_max = min(weight, y_center + y_len // 2)
        x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
        
        x_center_crop = x_center - x_min
        y_center_crop = y_center - y_min

        vm_crop = vm_no_crop[x_min:x_max+1, y_min:y_max+1, 0].astype(bool)
        fm_crop = fm_no_crop[x_min:x_max+1, y_min:y_max+1, 0].astype(bool)
        img_crop = img[x_min:x_max+1, y_min:y_max+1]
        vm_crop_gt = vm_no_crop_gt[x_min:x_max+1, y_min:y_max+1, 0].astype(bool)

        h, w = vm_crop.shape[:2]
        m = transform.rescale(vm_crop, (self.patch_h/h, self.patch_w/w))
        cur_h, cur_w = m.shape[:2]
        to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
        m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]
        vm_crop = m[np.newaxis, ...]

        center_crop = np.zeros_like(vm_crop[0])
        x_center_crop = int(x_center_crop*self.patch_h/h)
        y_center_crop = int(y_center_crop*self.patch_w/w)
        center_crop[x_center_crop: x_center_crop+1, y_center_crop: y_center_crop+1]=1
        center_crop = self.generate_heatmap(center_crop.astype(np.float), (35, 35), 9)
        center_crop = center_crop[np.newaxis, ...]

        img_ = transform.rescale(img_crop, (self.patch_h/h, self.patch_w/w, 1))
        cur_h, cur_w = img_.shape[:2]
        to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)), (0, 0))
        img_ = np.pad(img_, to_pad)[:self.patch_h, :self.patch_w, :3]
        img_crop = img_

        h, w = vm_crop_gt.shape[:2]
        m = transform.rescale(vm_crop_gt, (self.patch_h/h, self.patch_w/w))
        cur_h, cur_w = m.shape[:2]
        to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
        m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]
        vm_crop_gt = m[np.newaxis, ...]

        m = transform.rescale(fm_crop, (self.patch_h/h, self.patch_w/w))
        cur_h, cur_w = m.shape[:2]
        to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
        m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]    
        fm_crop = m[np.newaxis, ...]

        loss_mask = fm_no_crop.astype(int)-vm_no_crop_gt.astype(int)
        loss_mask[loss_mask==255]=0
        loss_mask = 1-loss_mask.astype(bool)

        vm_no_crop = vm_no_crop[np.newaxis, ...]
        fm_no_crop = fm_no_crop[np.newaxis, ...]

        obj_position = np.array([x_min, x_max, y_min, y_max])
        vm_pad = np.array([max(self.patch_h-cur_h, 0), max(self.patch_w-cur_w, 0)])
        vm_scale = np.array([self.patch_h/h, self.patch_w/w])
        counts = np.array([1])

        counts = torch.from_numpy(counts).to(self.dtype).to(self.device)

        obj_position = torch.from_numpy(obj_position).to(self.dtype).to(self.device)
        vm_pad = torch.from_numpy(vm_pad).to(self.dtype).to(self.device)
        vm_scale = torch.from_numpy(vm_scale).to(self.dtype).to(self.device)

        fm_crop = torch.from_numpy(fm_crop).to(self.dtype).to(self.device)
        fm_no_crop = torch.from_numpy(np.array(fm_no_crop)).to(self.dtype).to(self.device)
        vm_crop = torch.from_numpy(vm_crop).to(self.dtype).to(self.device)
        vm_crop_gt = torch.from_numpy(vm_crop_gt).to(self.dtype).to(self.device)
        vm_no_crop = torch.from_numpy(np.array(vm_no_crop)).to(self.dtype).to(self.device)
        center_crop = torch.from_numpy(np.array(center_crop)).to(self.dtype).to(self.device)
        
        img_crop = torch.from_numpy(np.array(img_crop)).to(self.dtype).to(self.device)
        img = torch.from_numpy(np.array(img)).to(self.dtype).to(self.device)

        loss_mask = torch.from_numpy(np.array(loss_mask)).to(self.dtype).to(self.device)
    
        image_id = torch.from_numpy(np.array(image_id)).to(self.dtype).to(self.device)
        anno_id = torch.from_numpy(np.array(anno_id)).to(self.dtype).to(self.device)
        occlude_rate = torch.from_numpy(np.array(occlude_rate)).to(self.dtype).to(self.device)
        
        if self.mode=="train":
            meta = {
                # "vm_no_crop": vm_no_crop,
                "vm_crop": vm_crop,
                # "vm_crop_gt": vm_crop_gt,
                # "fm_no_crop": fm_no_crop,
                "fm_crop": fm_crop,
                "img_crop": img_crop,
                "center_crop": center_crop,
                # "loss_mask": loss_mask,
                "obj_position": obj_position,
                "vm_pad": vm_pad,
                "vm_scale": vm_scale,
                "counts":counts,
                "img_id": image_id,
                "anno_id": anno_id,
                "img_no_crop": img,
            }
        elif self.mode=="test":
            meta = {
                "vm_no_crop": vm_no_crop,
                "vm_no_crop_gt": vm_no_crop_gt,
                "vm_crop": vm_crop,
                "vm_crop_gt": vm_crop_gt,
                "fm_no_crop": fm_no_crop,
                "fm_crop": fm_crop,
                "img_crop": img_crop,
                "center_crop": center_crop,
                "loss_mask": loss_mask,
                "obj_position": obj_position,
                "vm_pad": vm_pad,
                "vm_scale": vm_scale,
                "counts":counts,
                "img_id": image_id,
                "anno_id": anno_id,
                "occlude_rate":occlude_rate,
                # # for vq
                # "mask_crop": fm_crop,
                "img": img,
            }
        return meta

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

    def polys_to_mask(self, polygons, height, width):
        rles = mask_utils.frPyObjects(polygons, height, width)
        rle = mask_utils.merge(rles)
        mask = mask_utils.decode(rle)
        return mask

