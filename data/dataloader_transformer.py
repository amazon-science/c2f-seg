from data.dataloader_Fishbowl import FishBowl
from data.dataloader_MOViD_A import MOViD_A
from data.dataloader_KINS import Kins_Fusion_dataset, KINS_Aisformer_VRSP_Intersection
from data.dataloader_COCOA import COCOA_Fusion_dataset, COCOA_VRSP

def load_dataset(config, args, mode):
    if mode=="train":
        if args.dataset=="KINS":
            train_dataset = Kins_Fusion_dataset(config, mode='train')
            test_dataset = Kins_Fusion_dataset(config, mode='test')
        elif args.dataset=="COCOA":
            train_dataset = COCOA_Fusion_dataset(config, mode='train')
            test_dataset = COCOA_Fusion_dataset(config, mode='test')
        elif args.dataset=="Fishbowl":
            train_dataset = FishBowl(config, mode='train')
            test_dataset = FishBowl(config, mode='test')
        elif args.dataset=="MOViD_A":
            train_dataset = MOViD_A(config, mode='train')
            test_dataset = MOViD_A(config, mode='test')
        return train_dataset, test_dataset 
    else:
        if args.dataset=="KINS":
            test_dataset = KINS_Aisformer_VRSP_Intersection(config, mode='test')
        elif args.dataset=="COCOA":
            test_dataset = COCOA_Fusion_dataset(config, mode='test')
        elif args.dataset=="Fishbowl":
            test_dataset = FishBowl(config, mode='test')
        elif args.dataset=="MOViD_A":
            test_dataset = MOViD_A(config, mode='test')
        return test_dataset