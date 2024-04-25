import torch

import torch.nn.functional as F
import numpy as np
import sys
import tqdm
import os

import math
import matplotlib.pyplot as plt
from .networks import skip, fcn
from .SSIM import SSIM
import scipy.io as sio
import xlwt
from torch import optim
import torch.nn.utils as nutils
from scipy.signal import convolve2d

sys.path.append('../')
from .util import evaluation_image, get_noise, move2cpu, calculate_psnr, save_final_kernel_png, tensor2im01, calculate_parameters
from .kernel_generate import gen_kernel_random, gen_kernel_random_motion, make_gradient_filter, ekp_kernel_generator

sys.path.append('../../')

from torch.utils.tensorboard import SummaryWriter

'''
# ------------------------------------------
# models of DIPDKP, etc.
# ------------------------------------------
'''






class DIPDKP:
    '''
    # ------------------------------------------
    # (1) create model, loss and optimizer
    # ------------------------------------------
    '''

    def estimate_variance(self, padding_mode="reflect"):
        self.noise2 = (self.lr - self.blur_and_downsample.data)**2
        self.noise2_mean = self.noise2.mean()
        if self.conf.noise_estimator == 'niid':
            noise2_pad = F.pad(input=self.noise2, mode = padding_mode, pad = ((self.args.window_variance - 1) //2, )*4)
            self.lambda_p = F.conv2d(input = noise2_pad,
                                     weight = self.var_filter.expand(self.lr.shape[1], -1, -1, -1),
                                     groups= self.lr.shape[1])
        elif self.conf.noise_estimator == 'iid':
            self.lambda_p = torch.ones_like(self.lr) * self.noise2.mean()

        elif self.conf.noise_estimator == 'no-noise':
            self.lambda_p = torch.ones_like(self.lr)

        else:
            sys.exit('Please input corrected noise estimation methods: iid or niid!')

    def calculate_grad_abs(self, padding_mode="reflect"):
        hr_pad = F.pad(input = self.im_HR_est, mode = padding_mode, pad = (1, ) * 4)
        out = F.conv3d(input = hr_pad.expand(self.grad_filters.shape[0], -1, -1, -1).unsqueeze(0),
                       weight = self.grad_filters.unsqueeze(1).unsqueeze(1),
                       stride = 1, groups = self.grad_filters.shape[0])

        return torch.abs(out.squeeze(0))

    def initialize_K(self):

        # E Kernel Prior
        l1 = 1 / (self.sf * 1.00)
        self.kernel_code = torch.tensor([[l1,  0.0],
                                         [0.0, l1]], dtype=torch.float32).cuda()
        self.kernel_code.requires_grad = True
        self.optimizer_kernel = optim.Adam(params=[self.kernel_code,], lr=5e-3)

    def MCMC_sampling(self):

        # random Gaussian kernel
        if self.conf.model == 'DIPDKP':
            kernel_random = gen_kernel_random(self.k_size, self.sf, self.min_var, self.max_var, 0, self.conf.kernel_x,
                                              self.conf.kernel_y)

        # random linear motion kernel
        elif self.conf.model == 'DIPDKP-motion':
            lens = int((min(self.sf * 4 + 3, 21)) / 4)
            kernel_random = gen_kernel_random_motion(self.k_size, self.sf, lens, noise_level=0)

        # random motion kernel
        elif self.conf.model == 'DIPDKP-random-motion':
            num = len(os.listdir(self.conf.motion_blur_path)) // 2
            random_num = int(np.random.rand() * num)
            kernel_random = sio.loadmat(os.path.join(self.conf.motion_blur_path,
                                                     "MotionKernel_{}_{}".format(random_num, self.conf.jj)))[
                'Kernel']

        self.kernel_random = torch.from_numpy(kernel_random).type(torch.FloatTensor).to(
            torch.device('cuda')).unsqueeze(0).unsqueeze(0)

    def MC_warm_up(self):
        if (self.conf.model == 'DIPDKP' or self.conf.model == 'DIPDKP-motion' or self.conf.model == 'DIPDKP-random-motion'): # and iteration == 0
            # log_str = 'Number of parameters in Generator-x: {:.2f}K'
            # print(log_str.format(calculate_parameters(self.net_dip) / 1000))
            # log_str = 'Number of parameters in Generator-k: {:.2f}K'
            # print(log_str.format(calculate_parameters(self.net_kp) / 1000))

            for i in range(self.conf.kernel_first_iteration):
                kernel = self.net_kp(self.kernel_code).view(1, 1, self.kernel_size, self.kernel_size)

                self.MCMC_sampling()

                lossk = self.mse(self.kernel_random, kernel)

                # lossk = self.KLloss(kernel, kernel_random) + self.conf.jj_kl * self.KLloss(kernel_random, kernel)
                lossk.backward(retain_graph=True)
                lossk.detach()

                self.optimizer_kp.step()
                self.optimizer_kp.zero_grad()

    def print_and_output_setting(self):
        # print setting
        self.wb = xlwt.Workbook()  # 一个实例
        self.sheet = self.wb.add_sheet("Sheet1")  # 工作簿名称
        self.sheet.write(0, 1, "image PSNR")
        self.sheet.write(0, 2, "RE loss")
        self.sheet.write(0, 3, "kernel PSNR")
        for i in range(1, 1000):
            self.sheet.write(i, 0, str(i))  # 不设置格式
            self.wb.save(os.path.abspath(os.path.join(self.conf.output_dir_path, self.conf.img_name + '.xls')))  # 最后一定要保存，否则无效

        fold = self.conf.output_dir_path
        self.writer_model = SummaryWriter(log_dir=fold, flush_secs=20)

    def print_and_output(self, sr, kernel, kernel_gt, loss_x, i_p):
        save_final_kernel_png(move2cpu(kernel.squeeze()), self.conf, self.conf.kernel_gt,
                              (self.iteration * self.conf.I_loop_x + i_p))
        plt.imsave(os.path.join(self.conf.output_dir_path,
                                '{}_{}.png'.format(self.conf.img_name, (self.iteration * self.conf.I_loop_x + i_p))),
                   tensor2im01(sr), vmin=0, vmax=1., dpi=1)

        image_psnr, image_ssim = evaluation_image(self.hr, sr, self.sf)
        kernel_np = move2cpu(kernel.squeeze())
        kernel_psnr = calculate_psnr(kernel_gt, kernel_np, is_kernel=True)

        if self.conf.IF_print == True:
            print('\n Iter {}, loss: {}, PSNR: {}, SSIM: {}'.format(self.iteration, loss_x.data, image_psnr,
                                                                    image_ssim))

        self.writer_model.add_scalar('Image_PSNR/' + self.conf.img_name, image_psnr,
                                     (self.iteration * self.conf.I_loop_x + i_p))
        self.writer_model.add_scalar('RE_loss/' + self.conf.img_name, loss_x.data,
                                     (self.iteration * self.conf.I_loop_x + i_p))
        self.writer_model.add_scalar('Kernel_PSNR/' + self.conf.img_name, kernel_psnr,
                                     (self.iteration * self.conf.I_loop_x + i_p))

        black_style = xlwt.easyxf("font:colour_index black;") # set style

        self.sheet.write((self.iteration * self.conf.I_loop_x + i_p) + 1, 1, '%.2f' % image_psnr, black_style)
        self.wb.save(self.conf.output_dir_path + "/" + self.conf.img_name + '.xls')
        self.sheet.write((self.iteration * self.conf.I_loop_x + i_p) + 1, 2, '%.2f' % loss_x.data, black_style)
        self.wb.save(self.conf.output_dir_path + "/" + self.conf.img_name + '.xls')
        self.sheet.write((self.iteration * self.conf.I_loop_x + i_p) + 1, 3, '%.2f' % kernel_psnr, black_style)
        self.wb.save(self.conf.output_dir_path + "/" + self.conf.img_name + '.xls')

    def __init__(self, conf, lr, hr, device=torch.device('cuda')):

        # Acquire configuration
        self.conf = conf
        self.lr = lr
        self.sf = conf.sf
        self.hr = hr
        self.kernel_size = min(conf.sf * 4 + 3, 21)
        self.min_var = 0.175 * self.sf + self.conf.var_min_add
        self.max_var = min(2.5 * self.sf, 10) + self.conf.var_max_add
        self.k_size = np.array(
            [min(self.sf * 4 + 3, 21),
             min(self.sf * 4 + 3, 21)])  # 11x11, 15x15, 19x19, 21x21 for x2, x3, x4, x8

        # DIP model
        _, C, H, W = self.lr.size()
        self.input_dip = get_noise(C, 'noise', (H * self.sf, W * self.sf)).to(device).detach()
        self.lr_scaled = F.interpolate(self.lr, size=[H * self.sf, W * self.sf], mode='bicubic', align_corners=False)
        # self.input_dip = self.lr_scaled
        self.input_dip.requires_grad = False
        self.net_dip = skip(C, 3,
                            num_channels_down=[128, 128, 128, 128, 128],
                            num_channels_up=[128, 128, 128, 128, 128],
                            num_channels_skip=[16, 16, 16, 16, 16],
                            upsample_mode='bilinear',
                            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
        self.net_dip = self.net_dip.to(device)
        self.optimizer_dip = torch.optim.Adam([{'params': self.net_dip.parameters()}], lr=conf.dip_lr)

        # fc layers as kernel prior, accroding to Double-DIP/Selfdeblur, set lr = 1e-4
        if conf.model == 'DIPDKP' or conf.model == 'DIPDKP-motion' or conf.model == 'DIPDKP-random-motion':
            n_k = 200
            self.kernel_code = get_noise(n_k, 'noise', (1, 1)).detach().squeeze().to(device)
            self.kernel_code.requires_grad = False
            self.net_kp = fcn(n_k, self.kernel_size ** 2).to(device)
            self.optimizer_kp = torch.optim.Adam([{'params': self.net_kp.parameters()}], lr=conf.DIPDKP_kp_lr) #  1e-4

        # loss setting
        self.ssimloss = SSIM().to(device)
        self.mse = torch.nn.MSELoss().to(device)
        self.KLloss = torch.nn.KLDivLoss(reduction='mean').to(device)

        print('*' * 60 + '\nSTARTED {} on: {}...'.format(conf.model, conf.input_image_path))

        # noise prior
        self.grad_filters = make_gradient_filter()
        # self.make_gradient_filter()
        self.num_pixels = self.lr.numel()
        self.lambda_p = torch.ones_like(self.lr, requires_grad=False) * (0.01 ** 2)
        self.noise2_mean = 1

    '''
    # ---------------------
    # (2) training
    # ---------------------
    '''

    def train(self):

        self.print_and_output_setting()

        _, C, H, W = self.lr.size()

        path = os.path.join(self.conf.input_dir, self.conf.filename).replace('lr_x', 'gt_k_x').replace('.png', '.mat')
        if self.conf.real == False:
            kernel_gt = sio.loadmat(path)['Kernel']
        else:
            kernel_gt = np.zeros([self.kernel_size, self.kernel_size])

        self.MC_warm_up()

        for self.iteration in tqdm.tqdm(range(self.conf.max_iters), ncols=60):

            if self.conf.model == 'DIPDKP':
                # zero loss and gradient
                self.kernel_code.requires_grad = False
                self.optimizer_kp.zero_grad()
                # compute SR image
                sr = self.net_dip(self.input_dip)
                sr_pad = F.pad(sr, mode='circular',
                               pad=(self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2,
                                    self.kernel_size // 2))
                # initial loss
                k_losses = torch.zeros(self.conf.D_loop)
                k_loss_probability = torch.zeros(self.conf.D_loop)
                k_loss_weights = torch.zeros(self.conf.D_loop)
                x_losses = torch.zeros(self.conf.D_loop)

                for k_p in range(self.conf.D_loop):
                    kernel = self.net_kp(self.kernel_code).view(1, 1, self.kernel_size, self.kernel_size)
                    self.MCMC_sampling()
                    k_losses[k_p] = self.mse(self.kernel_random, kernel)
                    # k_losses[k_p] = self.KLloss(kernel, kernel_random) + self.conf.jj_kl * self.KLloss(kernel_random, kernel)
                    out_x = F.conv2d(sr_pad, self.kernel_random.expand(3, -1, -1, -1).clone().detach(), groups=3)
                    out_x = out_x[:, :, 0::self.sf, 0::self.sf]
                    x_losses[k_p] = self.mse(out_x, self.lr)

                sum_exp_x_losses = 1e-5
                lossk = 0
                for i in range(self.conf.D_loop):
                    # sum_exp_x_losses += math.exp(x_losses[i])
                    sum_exp_x_losses += (x_losses[i]-min(x_losses))

                for i in range(self.conf.D_loop):
                    k_loss_probability[i] = (x_losses[i]-min(x_losses))/sum_exp_x_losses
                    k_loss_weights[i] = (-(1 - k_loss_probability[i])**2) * torch.log(k_loss_probability[i]+1e-3)
                    # k_loss_weights[i] = k_loss_probability[i]
                    lossk += k_loss_weights[i].clone().detach() * k_losses[i]

                if self.conf.D_loop != 0:
                    lossk.backward(retain_graph=True)
                    lossk.detach()
                    self.optimizer_kp.step()

                ac_loss_k = 0
                for i_p in range(self.conf.I_loop_x):

                    # for p in self.net_kp.parameters(): p.requires_grad = False
                    # for p in self.net_dip.parameters(): p.requires_grad = True
                    # zero gradient
                    self.optimizer_dip.zero_grad()
                    self.optimizer_kp.zero_grad()

                    # generate kernel and image
                    kernel = self.net_kp(self.kernel_code).view(1, 1, self.kernel_size, self.kernel_size)
                    sr = self.net_dip(self.input_dip)

                    # compute data prior
                    sr_pad = F.pad(sr, mode='circular',
                                   pad=(self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2,
                                        self.kernel_size // 2))
                    out_x = F.conv2d(sr_pad, kernel.expand(3, -1, -1, -1).clone().detach(), groups=3)
                    out_x = out_x[:, :, 0::self.sf, 0::self.sf]

                    # adding image disturbance
                    disturb = np.random.normal(0, np.random.uniform(0, self.conf.Image_disturbance), out_x.shape)
                    disturb_tc = torch.from_numpy(disturb).type(torch.FloatTensor).to(
                        torch.device('cuda'))

                    # first use SSIM because it helps the model converge faster
                    if self.iteration <= 80:  #80 for DIPDKP
                        loss_x = 1 - self.ssimloss(out_x, self.lr + disturb_tc)
                    else:
                        loss_x = self.mse(out_x, self.lr + disturb_tc)

                    self.im_HR_est = sr
                    grad_loss = self.conf.grad_loss_lr * self.noise2_mean * 0.20 * torch.pow(
                        self.calculate_grad_abs() + 1e-8, 0.67).sum() / self.num_pixels

                    loss_x_update = loss_x + grad_loss

                    loss_x_update.backward(retain_graph=True)
                    loss_x_update.detach()

                    self.optimizer_dip.step()

                    out_k = F.conv2d(sr_pad.clone().detach(), kernel.expand(3, -1, -1, -1), groups=3)
                    out_k = out_k[:, :, 0::self.sf, 0::self.sf]

                    if self.iteration <= 80:  #80 for DIPDKP
                        loss_k = 1 - self.ssimloss(out_k, self.lr)
                    else:
                        loss_k = self.mse(out_k, self.lr)

                    ac_loss_k = ac_loss_k + loss_k
                    # loss_k.backward(retain_graph=True)
                    # loss_k.detach()
                    # self.optimizer_kp.step()

                    # update
                    if (self.iteration * self.conf.I_loop_x + i_p + 1) % (self.conf.I_loop_k) == 0:
                        # for p in self.net_kp.parameters(): p.requires_grad = True
                        # for p in self.net_dip.parameters(): p.requires_grad = False
                        ac_loss_k.backward(retain_graph=True)
                        self.optimizer_kp.step()
                        ac_loss_k = 0

                    # print and output
                    if (((self.iteration * self.conf.I_loop_x + i_p) + 1) % self.conf.Print_iteration == 0 or (
                            (self.iteration * self.conf.I_loop_x + i_p) + 1) == 1):
                        self.print_and_output(sr, kernel, kernel_gt, loss_x, i_p)

        kernel = move2cpu(kernel.squeeze())

        save_final_kernel_png(kernel, self.conf, self.conf.kernel_gt)

        if self.conf.verbose:
            print('{} estimation complete! (see --{}-- folder)\n'.format(self.conf.model,
                                                                         self.conf.output_dir_path) + '*' * 60 + '\n\n')

        return kernel, sr   # sr, self.lagd_y


class SphericalOptimizer(torch.optim.Optimizer):
    ''' spherical optimizer, optimizer on the sphere of the latent space'''

    def __init__(self, kernel_size, optimizer, params, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params
        with torch.no_grad():
            # in practice, setting the radii as kernel_size-1 is slightly better
            self.radii = {param: torch.ones([1, 1, 1]).to(param.device) * (kernel_size - 1) for param in params}

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        for param in self.params:
            param.data.div_((param.pow(2).sum(tuple(range(2, param.ndim)), keepdim=True) + 1e-9).sqrt())
            param.mul_(self.radii[param])

        return loss
