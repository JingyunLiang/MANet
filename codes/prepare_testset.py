import os.path
import logging
import argparse
import numpy as np
import torch
import sys
import options.options as option
from data import create_dataset, create_dataloader
import utils.util as util


def generate_dataset(opt, test_loader, save_dir_HR, save_dir_LR, device_id):
    prepro = util.SRMDPreprocessing(opt['scale'], random=False, l=opt['kernel_size'], add_noise=opt['test_noise'],
                                    noise_high=opt['noise'] / 255., rate_cln=-1,
                                    device=torch.device('cuda:{}'.format(device_id)), sig=opt['sig'], sig1=opt['sig1'],
                                    sig2=opt['sig2'], theta=opt['theta'],
                                    sig_min=opt['sig_min'], sig_max=opt['sig_max'], rate_iso=opt['rate_iso'],
                                    is_training=False, sv_mode=opt['sv_mode'])

    for test_data in test_loader:
        img_name = os.path.splitext(os.path.basename(test_data['GT_path'][0]))[0]
        test_data['GT'] = test_data['GT'].to(torch.device('cuda:{}'.format(device_id)))
        GT_img = util.tensor2img(test_data['GT'])
        LR_img, LR_n_img, ker_map, kernel = prepro(test_data['GT'], kernel=True)
        LR_n_img = util.tensor2img(LR_n_img)  # uint8

        # save images
        if opt['sv_mode'] == 0:
            img_name += '_{:.1f}_{:.1f}_{:.1f}.png'.format(opt['sig1'], opt['sig2'], opt['theta'])
            print('processing {:>30s} for scale {}, SI mode with kernel: '.format(img_name, opt['scale']), ker_map.cpu())
        else:
            img_name += '.png'
            print('processing {:>30s} for scale {}, SV mode {}'.format(img_name, opt['scale'], opt['sv_mode']))

        util.save_img(GT_img, os.path.join(save_dir_HR, img_name))
        util.save_img(LR_n_img, os.path.join(save_dir_LR, img_name))


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='options/test/prepare_testset.yml',
                        help='Path to options YMAL file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    #### mkdir and logger
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                 and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    device_id = torch.cuda.current_device()

    # set random seed
    util.set_random_seed(0)

    for scale in [2, 3, 4]:
        opt['scale'] = scale

        #### Create test dataset and dataloader
        test_loaders = []
        for phase, dataset_opt in sorted(opt['datasets'].items()):
            dataset_opt['scale'] = scale
            test_set = create_dataset(dataset_opt)
            test_loader = create_dataloader(test_set, dataset_opt)
            logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
            test_loaders.append(test_loader)

        for test_loader in test_loaders:
            test_set_name = test_loader.dataset.opt['name']
            logger.info('\nGenerating [{:s}]...'.format(test_set_name))

            for noise in [0, 15]:
                opt['noise'] = noise
                if noise > 0: opt['test_noise'] = True

                if opt['noise'] == 15 and opt['scale'] != 4:
                    continue

                sv_modes = [0, 1, 2, 3, 4, 5] if test_set_name == 'BSD100' else [0]
                for sv_mode in sv_modes:
                    opt['sv_mode'] = sv_mode

                    save_dir = test_loader.dataset.opt['dataroot_GT'].replace('/HR', '_x{}'.format(opt['scale']))
                    save_dir_HR = os.path.join(save_dir, 'HR_si') if opt['sv_mode'] == 0 else os.path.join(save_dir,
                                                                                                        'HR_sv')
                    save_dir_LR = os.path.join(save_dir, 'LR_mode{}_noise{}'.format(opt['sv_mode'], opt['noise']))
                    util.mkdir(save_dir_HR)
                    util.mkdir(save_dir_LR)

                    # spatial-invariant
                    if opt['sv_mode'] == 0:
                        for sig1 in [1, 1 + opt['scale'], 1 + 2 * opt['scale']]:
                            opt['sig1'] = sig1
                            for sig2 in range(1, 1 + sig1, opt['scale']):
                                opt['sig2'] = sig2
                                for theta in [0, np.pi / 4]:
                                    opt['theta'] = theta

                                    if sig1 == sig2 and theta > 0:
                                        continue

                                    generate_dataset(opt, test_loader, save_dir_HR, save_dir_LR, device_id)

                    # spatial-variant
                    else:
                        generate_dataset(opt, test_loader, save_dir_HR, save_dir_LR, device_id)


    print('\n \nNote: \nFor spatially invariant (SI) SR, HR and LR images are in `HR_si` and `LR_mode0_noise0`, respectively. \n'
          'For spatially variant (SV) SR, HR and LR images are organized as `HR_sv` and `LR_mode1_noise0`, respectively.\n\n')

if __name__ == '__main__':
    main()
    sys.exit(0)
