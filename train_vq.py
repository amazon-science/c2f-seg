import os
import cv2
import random
import numpy as np
import torch
import argparse
import time
from shutil import copyfile
from torch.utils.data import DataLoader
from data.dataloader_vqgan import load_dataset
from utils.evaluation import get_IoU
from utils.logger import setup_logger
from utils.utils import Config, Progbar, to_cuda, stitch_images
from utils.utils import get_lr_schedule_with_steps, torch_init_model
from taming_src.vqperceptual import VQLPIPSWithDiscriminator, adopt_weight
from taming_src.taming_models import VQModel

def restore(ckpt_file, g_model, d_model, g_opt, d_opt):
    torch_init_model(g_model, ckpt_file, "g_model")
    torch_init_model(d_model, ckpt_file, "d_model")
    saving = torch.load(ckpt_file, map_location='cpu')
#     if 'optimizer_states' in saving:
#         opt_state = saving['optimizer_states']
# #         print(opt_state[0])
#         g_opt.load_state_dict(opt_state[0])
#         d_opt.load_state_dict(opt_state[1])
    print(f"Restored from {ckpt_file}")
    return g_opt, d_opt


def save(g_model, d_model, m_path, prefix=None, g_opt=None, d_opt=None):
    if prefix is not None:
        save_path = m_path + "_{}.pth".format(prefix)
    else:
        save_path = m_path + ".pth"

    print('\nsaving {}...\n'.format(save_path))
    all_saving = {'g_model': g_model.state_dict(),
                  'd_model': d_model.state_dict(),
                  'optimizer_states': [g_opt.state_dict(), d_opt.state_dict()]}
    torch.save(all_saving, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='model checkpoints path')
    parser.add_argument('--finetune_path', type=str, required=False, default=None)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--learn_type', default="mask", type=str)
    parser.add_argument('--check_point_path', default="../check_points", type=str)
    parser.add_argument('--dataset', default="Kins", type=str)

    args = parser.parse_args()
    args.path = os.path.join(args.check_point_path, args.path)  
    config_path = os.path.join(args.path, 'vqgan_{}.yml'.format(args.dataset))

    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('configs/vqgan_{}.yml'.format(args.dataset), config_path)

    # load config file
    config = Config(config_path)
    config.path = args.path

    # cuda visble devices
    local_rank = 0

    log_file = 'log-{}.txt'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    if local_rank == 0:
        logger = setup_logger(os.path.join(args.path, 'logs'), logfile_name=log_file)
        for k in config._dict:
            logger.info("{}:{}".format(k, config._dict[k]))
    else:
        logger = None

    # save samples and eval pictures
    os.makedirs(os.path.join(args.path, 'samples'), exist_ok=True)
    # os.makedirs(os.path.join(args.path, 'eval'), exist_ok=True)

    # init device
    if torch.cuda.is_available():
        config.device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
    else:
        config.device = torch.device("cpu")

    cv2.setNumThreads(0)
    # initialize random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    train_dataset, val_dataset = load_dataset(args, config)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=8,
        drop_last=True,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=2,
        drop_last=False,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )
    sample_iterator = val_dataset.create_iterator(config.sample_size)

    g_model = VQModel(config).to(config.device)
    d_model = VQLPIPSWithDiscriminator(config.model['params']['lossconfig']['params']).to(config.device)

    g_opt = torch.optim.Adam(list(g_model.encoder.parameters()) +
                             list(g_model.decoder.parameters()) +
                             list(g_model.quantize.parameters()) +
                             list(g_model.quant_conv.parameters()) +
                             list(g_model.post_quant_conv.parameters()),
                             lr=float(config.g_lr), betas=(0.5, 0.9))
    d_opt = torch.optim.Adam(d_model.discriminator.parameters(),
                             lr=float(config.d_lr), betas=(0.5, 0.9))

    d_sche = get_lr_schedule_with_steps(config.decay_type,
                                        d_opt,
                                        drop_steps=config.drop_steps,
                                        gamma=config.drop_gamma)
    g_sche = get_lr_schedule_with_steps(config.decay_type,
                                        g_opt,
                                        drop_steps=config.drop_steps,
                                        gamma=config.drop_gamma)

    # restore
    if args.finetune_path is not None:
        g_opt, d_opt = restore(args.finetune_path, g_model, d_model, g_opt, d_opt)


    g_model = g_model
    d_model = d_model

    steps_per_epoch = len(train_dataset) // config.batch_size
    iteration = g_model.iteration
    epoch = iteration // steps_per_epoch
    if local_rank == 0:
        logger.info('Start from epoch:{}, iteration:{}'.format(epoch, iteration))

    keep_training = True
    best_score = {}
    while (keep_training):
        epoch += 1
        stateful_metrics = ['epoch', 'iter', 'g_lr']
        if local_rank == 0:
            progbar = Progbar(len(train_dataset), max_iters=steps_per_epoch,
                              width=20, stateful_metrics=stateful_metrics)
        else:
            progbar = None

        for items in train_loader:
            g_model.train()
            d_model.train()
            items = to_cuda(items, config.device)
            g_model.iteration += 1
            iteration = g_model.iteration
            xrec, qloss = g_model.forward(items['mask_crop'])
            # opt dis
            
            d_opt.zero_grad()
            d_loss = d_model.forward(qloss, items['mask_crop'], xrec, optimizer_idx=1,
                                    global_step=g_model.iteration,
                                    split="train")

            d_loss.backward()
            d_opt.step()

            # opt gen
            g_opt.zero_grad()
            nll_loss, gan_loss, codebook_loss = d_model.forward(qloss, items['mask_crop'], xrec,
                                                                optimizer_idx=0,
                                                                global_step=g_model.iteration,
                                                                split="train")

            disc_factor = adopt_weight(d_model.disc_factor,
                                       g_model.iteration, threshold=d_model.discriminator_iter_start)
            d_weight = d_model.calculate_adaptive_weight(nll_loss, gan_loss,
                                                               last_layer=g_model.get_last_layer())
            g_loss = nll_loss + d_weight * disc_factor * gan_loss + codebook_loss

            g_loss.backward()
            g_opt.step()

            d_sche.step()
            g_sche.step()

            logs = [("g_loss", g_loss.item()), ("d_loss", d_loss.item()), ("nll_loss", nll_loss.item()),
                    ("d_weight", d_weight.item()), ("gan_loss", d_weight.item() * disc_factor * gan_loss.item()),
                    ("codebook_loss", codebook_loss.item())]

            logs = [("epoch", epoch), ("iter", g_model.iteration),
                    ('g_lr', g_sche.get_lr()[0])] + logs
            if local_rank == 0:
                progbar.add(config.batch_size, values=logs)

            if iteration % config.log_iters == 0 and local_rank == 0:
                logger.debug(str(logs))

            if iteration % config.sample_iters == 0 and local_rank == 0:
                g_model.eval()
                with torch.no_grad():
                    items = next(sample_iterator)
                    items = to_cuda(items, config.device)
                    fake_img, _ = g_model(items['mask_crop'])

                    fake_img = fake_img.mean(dim=1, keepdim=True)
                    fake_img = torch.clamp(fake_img, min=0, max=1)
                    fake_img = (fake_img > 0.5).to(torch.int64)
                    IoU = get_IoU(fake_img.long(), items['mask_crop'].long())
                    show_results = []
                    show_results.append(fake_img.permute(0, 2, 3, 1))
                    images = stitch_images(items['mask_crop'].permute(0, 2, 3, 1), show_results, img_per_row=2, mode="L")
                    
                mIoU = IoU.mean()
                logger.info("\n mIoU: {}".format(mIoU.item()))
                sample_name = os.path.join(args.path, 'samples', str(iteration).zfill(7) + ".png")
                print('\tsaving sample {}\n'.format(sample_name))
                images.save(sample_name)

            if iteration % config.save_iters == 0 and local_rank == 0:
                save(g_model, d_model, g_model.m_path, prefix='{}'.format(str(iteration)), g_opt=g_opt, d_opt=d_opt)

            if iteration >= config.max_iters:
                keep_training = False
                break

    if local_rank == 0:
        logger.info('Best score: ' + str(best_score))