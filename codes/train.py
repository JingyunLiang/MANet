import os
import math
import argparse
import random
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler
from data.util import bgr2ycbcr

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model

import socket
import getpass

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':  # Return the name of start method used for starting processes
        mp.set_start_method('spawn', force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ['RANK'])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)  # Initializes the default distributed process group


def main():
    ###### MANet train ######
    #### setup options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='options/train/train_stage1.yml',
                        help='Path to option YMAL file of MANet.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gpu_ids_qsub', type=str, default=None)
    parser.add_argument('--slurm_job_id', type=str, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, args.gpu_ids_qsub, is_train=True)
    device_id = torch.cuda.current_device()

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    util.set_random_seed(seed)

    pca_matrix_path = opt['pca_path']
    if pca_matrix_path is not None:
        if not os.path.exists(pca_matrix_path):
            # create PCA matrix of enough kernel and save it, to ensure all kernel have same corresponding kernel maps
            batch_ker, _ = util.random_batch_kernel(batch=150000, l=opt['kernel_size'],
                                                    sig_min=opt['sig_min'], sig_max=opt['sig_max'],
                                                    rate_iso=opt['rate_iso'],
                                                    scale=opt['scale'], tensor=False)
            print('batch kernel shape: {}'.format(batch_ker.shape))
            b = np.size(batch_ker, 0)
            batch_ker = batch_ker.reshape((b, -1))
            pca_matrix = util.PCA(batch_ker, k=opt['code_length']).float()
            print('PCA matrix shape: {}'.format(pca_matrix.shape))
            torch.save(pca_matrix, pca_matrix_path)
            print('Save PCA matrix at: {}'.format(pca_matrix_path))

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info('{}@{}, GPU {}, Job_id {}, Job path {}'.format(getpass.getuser(), socket.gethostname(),
                                                                   opt['gpu_ids'], args.slurm_job_id, os.getcwd()))
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### init online degradation function
    prepro_train = util.SRMDPreprocessing(opt['scale'], random=True, l=opt['kernel_size'], add_noise=opt['train_noise'],
                                          noise_high=opt['noise_high'] / 255., add_jpeg=opt['train_jpeg'], jpeg_low=opt['jpeg_low'],
                                          rate_cln=-1, device=torch.device('cuda:{}'.format(device_id)), sig=opt['sig'],
                                          sig1=opt['sig1'], sig2=opt['sig2'], theta=opt['theta'],
                                          sig_min=opt['sig_min'], sig_max=opt['sig_max'], rate_iso=opt['rate_iso'],
                                          is_training=True, sv_mode=0)
    prepro_val = util.SRMDPreprocessing(opt['scale'], random=False, l=opt['kernel_size'], add_noise=opt['test_noise'],
                                        noise_high=opt['noise'], add_jpeg=opt['test_jpeg'], jpeg_low=opt['jpeg'],
                                        rate_cln=-1, device=torch.device('cuda:{}'.format(device_id)), sig=opt['sig'],
                                        sig1=opt['sig1'], sig2=opt['sig2'], theta=opt['theta'],
                                        sig_min=opt['sig_min'], sig_max=opt['sig_max'], rate_iso=opt['rate_iso'],
                                        is_training=False, sv_mode=0)

    #### training
    # mixed precision
    scaler = torch.cuda.amp.GradScaler()

    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            if train_data['LQ'].shape[2] == 1:
                train_data['GT'] = train_data['GT'].to(torch.device('cuda:{}'.format(device_id)))
                LR_img, LR_n_img, ker_map, kernel = prepro_train(train_data['GT'], kernel=True)
            else:
                LR_img, LR_n_img, ker_map, kernel = train_data['LQ'], torch.zeros(1, 1), torch.zeros(1, 1, 1)

            #### update learning rate, schedulers
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### training
            model.feed_data(train_data, LR_img, LR_n_img, ker_map, kernel)
            model.optimize_parameters(current_step, scaler)

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}:{:.4e} '.format(k, v)
                    # tensorboard logger, but sometimes cause dead
                    # if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    #     if rank <= 0:
                    #         tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation
            if (current_step % opt['train'][
                'val_freq'] == 0 or current_step == 5 or current_step == 1000) and rank <= 0:
                avg_psnr = 0.0
                avg_psnr_y = 0.0
                avg_psnr_k = 0.0
                avg_mae_n = 0.0
                avg_lr_psnr_y = 0.0
                idx = 0
                for _, val_data in enumerate(val_loader):
                    idx += 1

                    val_data['GT'] = val_data['GT'].to(torch.device('cuda:{}'.format(device_id)))
                    LR_img, LR_n_img, ker_map, kernel = prepro_val(val_data['GT'], kernel=True)

                    model.feed_data(val_data, LR_img, LR_n_img, ker_map, kernel)
                    model.test()

                    visuals = model.get_current_visuals()

                    # Save SR images for reference
                    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], str(current_step))
                    util.mkdir(img_dir)
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))

                    # deal with kernel
                    save_ker_path = os.path.join(img_dir, '0kernel_{:s}_{:d}.png'.format(img_name, current_step))
                    if len(visuals['KE'].shape) > 2:
                        est_ker = util.tensor2img(visuals['KE'][5000, :, :], np.float32)
                    else:
                        est_ker = util.tensor2img(visuals['KE'], np.float32)
                    gt_ker = util.tensor2img(visuals['K'], np.float32)
                    util.plot_kernel(est_ker, save_ker_path, gt_ker)
                    avg_psnr_k += util.calculate_kernel_psnr(est_ker, gt_ker)

                    # calculate PSNR for LR
                    gt_img_lr = util.tensor2img(visuals['LQ'])
                    sr_img_lr = util.tensor2img(visuals['LQE'])
                    gt_img_lr = gt_img_lr / 255.
                    sr_img_lr = sr_img_lr / 255.

                    crop_border = opt['crop_border'] if opt['crop_border'] else opt['scale']
                    if gt_img_lr.shape[2] == 3:  # RGB image
                        sr_img_lr_y = bgr2ycbcr(sr_img_lr, only_y=True)
                        gt_img_lr_y = bgr2ycbcr(gt_img_lr, only_y=True)
                        if crop_border == 0:
                            cropped_sr_img_lr_y = sr_img_lr_y
                            cropped_gt_img_lr_y = gt_img_lr_y
                        else:
                            cropped_sr_img_lr_y = sr_img_lr_y[crop_border:-crop_border, crop_border:-crop_border]
                            cropped_gt_img_lr_y = gt_img_lr_y[crop_border:-crop_border, crop_border:-crop_border]
                        avg_lr_psnr_y += util.calculate_psnr(cropped_sr_img_lr_y * 255, cropped_gt_img_lr_y * 255)

                    # deal with sr image
                    sr_img = util.tensor2img(visuals['SR'])  # uint8
                    gt_img = util.tensor2img(visuals['GT'])  # uint8

                    util.save_img(sr_img, save_img_path)

                    # calculate PSNR
                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.

                    crop_size = opt['crop_border'] if opt['crop_border'] else opt['scale']
                    if crop_size == 0:
                        cropped_sr_img = sr_img
                        cropped_gt_img = gt_img
                    else:
                        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)

                    if gt_img.shape[2] == 3:  # RGB image
                        sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                        gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                        if crop_size == 0:
                            cropped_sr_img_y = sr_img_y
                            cropped_gt_img_y = gt_img_y
                        else:
                            cropped_sr_img_y = sr_img_y[crop_size:-crop_size, crop_size:-crop_size]
                            cropped_gt_img_y = gt_img_y[crop_size:-crop_size, crop_size:-crop_size]
                        avg_psnr_y += util.calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255)

                avg_psnr = avg_psnr / idx
                avg_psnr_y = avg_psnr_y / idx
                avg_psnr_k = avg_psnr_k / idx
                avg_mae_n = avg_mae_n / idx
                avg_lr_psnr_y = avg_lr_psnr_y / idx

                # log
                logger.info('# {}, Validation # PSNR_Y: {:.4f},  LR_PSNR_Y: {:.4f},  PSNR_K: {:.4f}'.format(
                        opt['name'], avg_psnr_y, avg_lr_psnr_y, avg_psnr_k))
                logger.info('{}@{}, GPU {}'.format(getpass.getuser(), socket.gethostname(), opt['gpu_ids']))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr_y: {:.6f}, lr_psnr_y: {:.6f}, psnr_k: {:.6f}'.format(
                        epoch, current_step, avg_psnr_y, avg_lr_psnr_y, avg_psnr_k))


                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('psnr_y', avg_psnr_y, current_step)
                    tb_logger.add_scalar('psnr_k', avg_psnr_k, current_step)
                    tb_logger.add_scalar('mae_n', avg_mae_n, current_step)
                    tb_logger.add_scalar('lr_psnr_y', avg_lr_psnr_y, current_step)

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of MANet training.')


if __name__ == '__main__':
    main()
