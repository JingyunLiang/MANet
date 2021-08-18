import torch
import logging
import models.modules.discriminator_vgg_arch as SRGAN_arch
import models.modules.MANet_arch as MANet_arch

logger = logging.getLogger('base')


####################
# define network
####################
#### Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'MANet_s1':
        netG = MANet_arch.MANet_s1(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                   nf=opt_net['nf'], nb=opt_net['nb'], gc=opt_net['gc'],
                                   scale=opt['scale'], pca_path=opt['pca_path'], code_length=opt['code_length'],
                                   kernel_size=opt['kernel_size'],
                                   manet_nf=opt_net['manet_nf'], manet_nb=opt_net['manet_nb'], split=opt_net['split'])
    elif which_model == 'MANet_s2':
        netG = MANet_arch.MANet_s2(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                   nf=opt_net['nf'], nb=opt_net['nb'], gc=opt_net['gc'],
                                   scale=opt['scale'], pca_path=opt['pca_path'], code_length=opt['code_length'],
                                   kernel_size=opt['kernel_size'],
                                   manet_nf=opt_net['manet_nf'], manet_nb=opt_net['manet_nb'], split=opt_net['split'])
    elif which_model == 'MANet_s3':
        netG = MANet_arch.MANet_s3(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                   nf=opt_net['nf'], nb=opt_net['nb'], gc=opt_net['gc'],
                                   scale=opt['scale'], pca_path=opt['pca_path'], code_length=opt['code_length'],
                                   kernel_size=opt['kernel_size'],
                                   manet_nf=opt_net['manet_nf'], manet_nb=opt_net['manet_nb'], split=opt_net['split'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG


# functions below are not used

#### Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


#### Define Network used for Perceptual Loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
