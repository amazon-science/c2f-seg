import os
import cv2
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from shutil import copyfile
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from data.dataloader_transformer import load_dataset
from utils.logger import setup_logger
from utils.utils import Config, to_cuda


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42) 
    # path
    parser.add_argument('--path', type=str, required=True, help='model checkpoints path')
    parser.add_argument('--check_point_path', type=str, default="../check_points", )
    parser.add_argument('--vq_path', type=str, required=True, default='KINS_vqgan')

    # dataset 
    parser.add_argument('--dataset', type=str, default="MOViD_A", help = "select dataset")
    parser.add_argument('--data_type', type=str, default="image", help = "select image or video model")
    parser.add_argument('--batch', type=int, default=1)

    parser.add_argument("--local_rank", default=-1, type=int, help="node rank for distributed training")

    args = parser.parse_args()

    if args.data_type=="image":
        from src.image_model import C2F_Seg
    elif args.data_type=="video":
        from src.video_model import C2F_Seg   
    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    rank = dist.get_rank()

    args.path = os.path.join(args.check_point_path, args.path)
    vq_model_path = os.path.join(args.check_point_path, args.vq_path)
    os.makedirs(args.path, exist_ok=True)

    config_path = os.path.join(args.path, 'c2f_seg_{}.yml'.format(args.dataset))
    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./configs/c2f_seg_{}.yml'.format(args.dataset), config_path)
    
    # load config file
    config = Config(config_path)
    config.path = args.path
    config.batch_size = args.batch
    config.dataset = args.dataset

    log_file = 'log-{}.txt'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger = setup_logger(os.path.join(args.path, 'logs'), logfile_name=log_file)
    if rank==0:
        # copy config template if does't exist
        if not os.path.exists(config_path):
            copyfile('./configs/c2f_seg_{}.yml'.format(args.dataset), config_path)

        # save samples and eval pictures
        os.makedirs(os.path.join(args.path, 'test_samples'), exist_ok=True)

        for k in config._dict:
            logger.info("{}:{}".format(k, config._dict[k]))

    # init device
    if torch.cuda.is_available():
        config.device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
    else:
        config.device = torch.device("cpu")
    n_gpu = torch.cuda.device_count()

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)
    # initialize random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    test_dataset = load_dataset(config, args, "test")
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(
        dataset=test_dataset,
        sampler=test_sampler,
        batch_size=config.batch_size,
        num_workers=8,
        drop_last=False
    )

    sample_iterator = test_dataset.create_iterator(config.sample_size)

    model = C2F_Seg(config, vq_model_path, mode='test', logger=logger)
    model.load(is_test=True ,prefix = config.stage2_iteration)
    model.restore_from_stage1(prefix = config.stage1_iteration)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    iter = 0
    iou = 0
    iou_count = 0
    invisible_iou_ = 0
    occ_count = 0

    iou_post = 0
    iou_count_post = 0
    invisible_iou_post = 0
    occ_count_post = 0

    model.eval()
    with torch.no_grad():
        if rank==0:
            test_loader = tqdm(test_loader)
        for items in test_loader:
            items = to_cuda(items, config.device)
            loss_eval = model.module.batch_predict_maskgit(items, iter, 'test', T=3)
            iter += 1
            iou += loss_eval['iou']
            iou_post += loss_eval['iou_post']
            iou_count += loss_eval['iou_count']
            invisible_iou_ += loss_eval['invisible_iou_']
            invisible_iou_post += loss_eval['invisible_iou_post']
            occ_count += loss_eval['occ_count']

            logger.info('Rank {}, iter {}: iou: {}, iou_post: {}, occ: {}, occ_post: {}'.format(
                rank, 
                iter-1,
                loss_eval['iou'].item(),
                loss_eval['iou_post'].item(),
                loss_eval['invisible_iou_'].item(),
                loss_eval['invisible_iou_post'].item(),
            ))
            dist.barrier()
            torch.cuda.empty_cache()
    dist.all_reduce(iou)
    dist.all_reduce(invisible_iou_)
    dist.all_reduce(iou_post)
    dist.all_reduce(invisible_iou_post)
    dist.all_reduce(iou_count)
    dist.all_reduce(occ_count)
    dist.barrier()
    if rank==0:
        logger.info('meanIoU: {}'.format(iou.item() / iou_count.item()))
        logger.info('meanIoU post-process: {}'.format(iou_post.item() / iou_count.item()))
        logger.info('meanIoU invisible: {}'.format(invisible_iou_.item() / occ_count.item()))
        logger.info('meanIoU invisible post-process: {}'.format(invisible_iou_post.item() / occ_count.item()))
        logger.info('iou_count: {}'.format(iou_count))
        logger.info('occ_count: {}'.format(occ_count))
    
