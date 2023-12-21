from data.dataloader_Fishbowl import FishBowl_VQ_Dataset
from data.dataloader_KINS import KINS_VQ_dataset
from data.dataloader_MOViD_A import Movid_A_VQ_Dataset
from data.dataloader_COCOA import COCOA_VQ_dataset

def load_dataset(args, config):
    if args.dataset=="KINS":
        train_dataset = KINS_VQ_dataset(config, mode='train')
        val_dataset = KINS_VQ_dataset(config, mode='test')
    elif args.dataset=="MOViD_A":
        train_dataset = Movid_A_VQ_Dataset(config, mode="train")
        val_dataset = Movid_A_VQ_Dataset(config, mode="test")
    elif args.dataset=="COCOA":
        train_dataset = COCOA_VQ_dataset(config, mode="train")
        val_dataset = COCOA_VQ_dataset(config, mode="test")
    elif args.dataset=="Fishbowl":
        train_dataset = FishBowl_VQ_Dataset(config, mode="train")
        val_dataset = FishBowl_VQ_Dataset(config, mode="test")
    return train_dataset, val_dataset