import os.path
import logging
import time
import argparse
from collections import OrderedDict
import numpy as np
import torch
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

#### options
parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, default='options/test/test_stage1.yml', help='Path to options YMAL file.')
parser.add_argument('--save_kernel', action='store_true', default=False, help='Save Kernel Esimtation.')
args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)
opt = option.dict_to_nonedict(opt)
device_id = torch.cuda.current_device()

#### mkdir and logger
util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
             and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

# set random seed
util.set_random_seed(0)

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']  # path opt['']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnr_k'] = []
    test_results['mae_n'] = []
    test_results['lr_psnr_y'] = []
    test_results['lr_ssim_y'] = []

    #### preprocessing for LR_img and kernel map
    prepro = util.SRMDPreprocessing(opt['scale'], random=False, l=opt['kernel_size'], add_noise=opt['test_noise'],
                                    noise_high=opt['noise'] / 255., add_jpeg=opt['test_jpeg'], jpeg_low=opt['jpeg'],
                                    rate_cln=-1, device=torch.device('cuda:{}'.format(device_id)), sig=opt['sig'],
                                    sig1=opt['sig1'], sig2=opt['sig2'], theta=opt['theta'],
                                    sig_min=opt['sig_min'], sig_max=opt['sig_max'], rate_iso=opt['rate_iso'],
                                    is_training=False, sv_mode=opt['sv_mode'])

    for test_data in test_loader:
        real_image = True if test_loader.dataset.opt['dataroot_GT'] is None else False
        generate_online = True if test_loader.dataset.opt['dataroot_GT'] is not None and test_loader.dataset.opt[
            'dataroot_LQ'] is None else False
        img_path = test_data['LQ_path'][0] if real_image else test_data['GT_path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        if real_image:
            LR_img, LR_n_img, ker_map, kernel = test_data['LQ'], test_data['LQ'], torch.ones(1, 1, 1), \
                                                torch.ones(1, 1, opt['kernel_size'], opt['kernel_size'])
        elif generate_online:
            test_data['GT'] = test_data['GT'].to(torch.device('cuda:{}'.format(device_id)))
            LR_img, LR_n_img, ker_map, kernel = prepro(test_data['GT'], kernel=True)
            print(ker_map.cpu())
        else:
            # note that it is not sutible for non-blind testing! because kernel is zero by default
            LR_img, LR_n_img, ker_map, kernel = test_data['LQ'], test_data['LQ'], torch.ones(1, 1, 1), \
                                                torch.ones(1, 1, opt['kernel_size'], opt['kernel_size'])

        model.feed_data(test_data, LR_img, LR_n_img, ker_map, kernel)
        model.test()

        visuals = model.get_current_visuals()

        sr_img = util.tensor2img(visuals['SR'])  # uint8

        # deal with the image margins for real images
        if test_loader.dataset.opt['mode'] == 'LQ':
            if opt['scale'] == 4:
                real_crop = 3
            elif opt['scale'] == 2:
                real_crop = 6
            elif opt['scale'] == 1:
                real_crop = 11
            assert real_crop * opt['scale'] * 2 > opt['kernel_size']
            sr_img = sr_img[real_crop * opt['scale']:-real_crop * opt['scale'],
                     real_crop * opt['scale']:-real_crop * opt['scale'], :]

        # save images
        suffix = opt['suffix']
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + '.png')
            save_ker_path = os.path.join(dataset_dir, '0kernel_{:s}{}.png'.format(img_name, suffix))
            save_ker_SV_path = os.path.join(dataset_dir, 'npz', img_name + suffix + '.npz')
        else:
            save_img_path = os.path.join(dataset_dir, img_name + '.png')
            save_ker_path = os.path.join(dataset_dir, '0kernel_{:s}.png'.format(img_name))
            save_ker_SV_path = os.path.join(dataset_dir, 'npz', img_name + '.npz')
        util.save_img(sr_img, save_img_path)
        if args.save_kernel:
            os.makedirs(os.path.join(dataset_dir, 'npz'), exist_ok=True)

        # choose a kernel to visualize from SV kernels
        if len(visuals['KE'].shape) > 2:
            est_ker = util.tensor2img(visuals['KE'][300, :, :], np.float32)
            est_ker_sv = visuals['KE'].float().cpu().numpy().astype(np.float32)
        else:
            est_ker = util.tensor2img(visuals['KE'], np.float32)
            est_ker_sv = None

        if real_image:
            util.plot_kernel(est_ker, save_ker_path)
            if args.save_kernel and est_ker_sv is not None:
                np.savez(save_ker_SV_path, sr_img=sr_img, est_ker_sv=est_ker_sv, gt_ker=0)

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
            lr_psnr_y = util.calculate_psnr(cropped_sr_img_lr_y * 255, cropped_gt_img_lr_y * 255)
            lr_ssim_y = util.calculate_ssim(cropped_sr_img_lr_y * 255, cropped_gt_img_lr_y * 255)
            test_results['lr_psnr_y'].append(lr_psnr_y)
            test_results['lr_ssim_y'].append(lr_ssim_y)

        # calculate PSNR and SSIM
        if not real_image:
            gt_ker = util.tensor2img(visuals['K'], np.float32)

            # for debug and visualization
            if not gt_ker.shape == est_ker.shape:
                gt_ker = est_ker

            util.plot_kernel(est_ker, save_ker_path, gt_ker)
            if args.save_kernel and est_ker_sv is not None:
                np.savez(save_ker_SV_path, sr_img=sr_img, est_ker_sv=est_ker_sv, gt_ker=gt_ker)
            psnr_k = util.calculate_kernel_psnr(est_ker, gt_ker)
            test_results['psnr_k'].append(psnr_k)

            gt_img = util.tensor2img(visuals['GT'])
            gt_img = gt_img / 255.
            sr_img = sr_img / 255.

            crop_border = opt['crop_border'] if opt['crop_border'] else opt['scale']
            if crop_border == 0:
                cropped_sr_img = sr_img
                cropped_gt_img = gt_img
            else:
                cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
                cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]

            psnr = 0  # util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            ssim = 0  # util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)

            if gt_img.shape[2] == 3:  # RGB image
                sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                if crop_border == 0:
                    cropped_sr_img_y = sr_img_y
                    cropped_gt_img_y = gt_img_y
                else:
                    cropped_sr_img_y = sr_img_y[crop_border:-crop_border, crop_border:-crop_border]
                    cropped_gt_img_y = gt_img_y[crop_border:-crop_border, crop_border:-crop_border]
                psnr_y = util.calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                ssim_y = util.calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
                logger.info(
                    '{:20s} - PSNR/SSIM: {:.2f}/{:.4f}; PSNR_Y/SSIM_Y: {:.2f}/{:.4f}; LR_PSNR_Y/LR_SSIM_Y: {:.2f}/{'
                    ':.4f}; PSNR_K: {:.2f} dB.'.format(
                        img_name, psnr, ssim, psnr_y, ssim_y, lr_psnr_y, lr_ssim_y, psnr_k))
            else:
                logger.info(
                    '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_K: {:.6f} dB.'.format(img_name, psnr, ssim, psnr_k))
        else:
            logger.info('{:20s} - LR_PSNR_Y/LR_SSIM_Y: {:.2f}/{:.4f}'.format(img_name, lr_psnr_y, lr_ssim_y))

    ave_lr_psnr_y = sum(test_results['lr_psnr_y']) / len(test_results['lr_psnr_y'])
    ave_lr_ssim_y = sum(test_results['lr_ssim_y']) / len(test_results['lr_ssim_y'])
    if not real_image:  # metrics
        # Average PSNR/SSIM results
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])

        if test_results['psnr_y'] and test_results['ssim_y']:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])

            ave_psnr_k = sum(test_results['psnr_k']) / len(test_results['psnr_k'])
            logger.info(
                '----{} ({} images), average PSNR_Y/SSIM_Y: {:.2f}/{:.4f}, LR_PSNR_Y/LR_SSIM_Y: {:.2f}/{:.4f}, '
                'kernel PSNR: {:.2f}\n'.
                    format(test_set_name, len(test_results['psnr_y']), ave_psnr_y, ave_ssim_y, ave_lr_psnr_y,
                           ave_lr_ssim_y, ave_psnr_k))

    else:
        logger.info('LR PSNR_K/LR_SSIM_Y: {:.2f}/{:.4f}\n'.format(ave_lr_psnr_y, ave_lr_ssim_y))
