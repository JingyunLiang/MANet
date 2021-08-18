# base model for blind SR, input LR, output kernel + SR
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.cuda.amp import autocast as autocast
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import CharbonnierLoss
import utils.util as util

logger = logging.getLogger('base')


class B_Model(BaseModel):
    def __init__(self, opt):
        super(B_Model, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()
        self.load_K()  # load the kernel estimation part

        # degradation model
        self.degradation_model = DegradationModel(opt['kernel_size'], opt['scale'], sv_degradation=True)
        self.degradation_model = nn.DataParallel(self.degradation_model)
        self.cal_lr_psnr = opt['cal_lr_psnr']

        if self.is_train:
            train_opt = opt['train']
            self.netG.train()

            # HR loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type is None:
                self.cri_pix = None
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # LR loss
            loss_type = train_opt['pixel_criterion_lr']
            if loss_type == 'l1':
                self.cri_pix_lr = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix_lr = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix_lr = CharbonnierLoss().to(self.device)
            elif loss_type is None:
                self.cri_pix_lr = None
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w_lr = train_opt['pixel_weight_lr']

            # kernel loss
            loss_type = train_opt['kernel_criterion']
            if loss_type == 'l1':
                self.cri_ker = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_ker = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_ker = CharbonnierLoss().to(self.device)
            elif loss_type is None:
                self.cri_ker = None
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_ker_w = train_opt['kernel_weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                print('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def init_model(self, scale=0.1):
        # Common practise for initialization.
        for layer in self.netG.modules():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_in')
                layer.weight.data *= scale  # for residual block
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_in')
                layer.weight.data *= scale
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                init.constant_(layer.weight, 1)
                init.constant_(layer.bias.data, 0.0)

    def feed_data(self, data, LR_img, LR_n_img, ker_map, kernel):
        self.real_H = data['GT'].to(self.device)  # GT
        self.var_L, self.var_LN, self.ker_map, self.real_K = LR_img.to(self.device), LR_n_img.to(
            self.device), ker_map.to(self.device), kernel.to(self.device)

    def optimize_parameters(self, step, scaler):
        self.optimizer_G.zero_grad()

        with autocast():
            l_all = 0
            self.fake_SR, self.fake_K = self.netG(self.var_LN, self.real_K)

            # hr loss
            if self.cri_pix is not None:
                l_pix = self.l_pix_w * self.cri_pix(self.fake_SR, self.real_H)
                l_all += l_pix
                self.log_dict['l_pix'] = l_pix.item()

            # kernel loss
            if self.cri_ker is not None:
                # times 1e4 since kernel pixel values are very small
                l_ker = self.l_ker_w * self.cri_ker(self.fake_K * 10000,
                                                    self.real_K.unsqueeze(1).expand(-1, self.fake_K.size(1), -1,
                                                                                    -1) * 10000) / self.fake_K.size(1)
                l_all += l_ker
                self.log_dict['l_ker'] = l_ker.item()

            # lr loss
            if self.cri_pix_lr is not None:
                self.fake_LR = self.degradation_model(self.real_H, self.fake_K)
                l_pix_lr = self.l_pix_w_lr * self.cri_pix_lr(self.fake_LR, self.var_L) # we should use LR before noise corruption as a ref
                l_all += l_pix_lr
                self.log_dict['l_pix_lr'] = l_pix_lr.item()

            else:
                self.fake_LR = self.var_L

        scaler.scale(l_all).backward()
        scaler.step(self.optimizer_G)
        scaler.update()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_SR, self.fake_K = self.netG(self.var_LN, self.real_K)

            if self.cal_lr_psnr:
                # synthesized data
                if self.real_H.shape[2] * self.real_H.shape[3] > 1:
                    self.fake_LR = self.degradation_model(self.real_H, self.fake_K)
                # no HR
                else:
                    self.fake_LR = self.degradation_model(self.fake_SR, self.fake_K)
            else:
                self.fake_LR = self.var_L

        self.netG.train()

    def test_x8(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netG.eval()

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [self.var_LN]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        with torch.no_grad():
            sr_list = [self.netG(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.fake_SR = output_cat.mean(dim=0, keepdim=True)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LQN'] = self.var_LN.detach()[0].float().cpu()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['LQE'] = self.fake_LR.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_SR.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        out_dict['ker_map'] = self.ker_map.detach()[0].float().cpu()
        out_dict['KE'] = self.fake_K.detach()[0].float().cpu()
        out_dict['K'] = self.real_K.detach()[0].float().cpu()
        out_dict['Batch_SR'] = self.fake_SR.detach().float().cpu()  # Batch SR, for train
        out_dict['Batch_KE'] = self.fake_K.detach().float().cpu()  # Batch SR, for train
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def load_K(self):
        load_path_K = self.opt['path']['pretrain_model_K']
        if load_path_K is not None:
            logger.info('Loading model for K [{:s}] ...'.format(load_path_K))
            self.load_network(load_path_K, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)


class DegradationModel(nn.Module):
    def __init__(self, kernel_size=15, scale=4, sv_degradation=True):
        super(DegradationModel, self).__init__()
        if sv_degradation:
            self.blur_layer = util.BatchBlur_SV(l=kernel_size, padmode='replication')
            self.sample_layer = util.BatchSubsample(scale=scale)
        else:
            self.blur_layer = util.BatchBlur(l=kernel_size, padmode='replication')
            self.sample_layer = util.BatchSubsample(scale=scale)

    def forward(self, image, kernel):
        return self.sample_layer(self.blur_layer(image, kernel))
