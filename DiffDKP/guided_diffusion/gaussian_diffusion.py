# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
import xlwt
import scipy.io as sio
import enum
import cv2
import numpy as np
import torch as th
import torch
import os
from PIL import Image
import torch.nn.functional as F
from collections import defaultdict

from .scheduler import get_schedule_jump

from .utils2 import *

from tqdm.auto import tqdm

import math
from scipy import fftpack

from .networks import skip, fcn, tiny_skip
from .SSIM import SSIM
import matplotlib.pyplot as plt

from .ComplexUnet import ComplexFCN, Complex_MSELoss

from  torchvision import utils as vutils
from thop import profile
from torchstat import stat

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, use_scale):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.

        if use_scale:
            scale = 1000 / num_diffusion_timesteps
        else:
            scale = 1

        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL



class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        conf=None
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        self.conf = conf

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_prev_prev = np.append(
            1.0, self.alphas_cumprod_prev[:-1])

        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = np.sqrt(self.alphas_cumprod_prev)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(
            1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def undo(self, image_before_step, img_after_model, est_x_0, t, debug=False):
        return self._undo(img_after_model, t)

    def _undo(self, img_out, t):
        beta = _extract_into_tensor(self.betas, t, img_out.shape)

        img_in_est = th.sqrt(1 - beta) * img_out + \
            th.sqrt(beta) * th.randn_like(img_out)

        return img_in_est

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1,
                                 t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2,
                                   t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(
            self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'x0_t': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        # stat(model, (1, 3, 256, 256))
        # flops, params = profile(model=model,
        #                         inputs=(torch.randn(1, 3, 256, 256), self._scale_timesteps(t),))

        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, model_var_values = th.split(model_output, C, dim=1)

        if self.model_var_type == ModelVarType.LEARNED:
            model_log_variance = model_var_values
            model_variance = th.exp(model_log_variance)
        else:
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x
        


        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            x0_t = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                x0_t = process_xstart(model_output)
            else:
                x0_t = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=x0_t, x_t=x, t=t
            )
            

            ## DDNM core ##


            if x0_t is not None:

                A, Ap = model_kwargs['A'], model_kwargs['Ap']
                sigma_y = model_kwargs['sigma_y']
                y = model_kwargs['y_patch']
                Apy = model_kwargs['Apy_patch']
                # y = model_kwargs['y_temp']
                # kernel = model_kwargs['kernel_tensor']
                xgt = model_kwargs['Xgt_patch']

                sigma_t = th.sqrt(_extract_into_tensor(self.posterior_variance, t, x0_t.shape))[0][0][0][0]
                a_t = _extract_into_tensor(self.posterior_mean_coef1, t, x0_t.shape)[0][0][0][0]

                # Eq. 19
                if sigma_t >= a_t*sigma_y:
                    lambda_t = 1
                    gamma_t = _extract_into_tensor(self.posterior_variance, t, x0_t.shape) - (a_t*lambda_t*sigma_y)**2
                else:
                    lambda_t = sigma_t/a_t*sigma_y
                    gamma_t = 0.

                # Eq. 17
                # x0_t = x0_t + lambda_t*self.Upsample(self.Y_small_patch-self.Downsample(x0_t))
                # x0_t_hat = x0_t - lambda_t*Ap(A(x0_t, self.K_est)-y, self.K_est)


                # Eq. FFT close-form function

                self.obtain_DKP_kernel()
                # self.K_est = self.K_gt

                beta = 1
                alph = self.alph_beta

                # x0_t_moved = Move_X_backward(x0_t)
                x0_t_moved = x0_t
                y_moved = Move_X_backward(y)
                # y_moved = y
                #
                #
                #
                k_ft = fft2(self.K_est, 256, 256)
                y_ft = fft2(y_moved)
                x0_t_ft = fft2(x0_t_moved)
                k_ft_c = torch.conj(k_ft)
                #
                up = beta * (k_ft_c * y_ft) + alph * x0_t_ft
                down = beta * (k_ft_c * k_ft) + alph

                x0_t_hat = ifft2(up/down)

                    # update kernel network via LR image reconstruction error
                if t[0] < 90:
                    for XK in range(self.XK):
                        # self.obtain_DKP_kernel()
                        self.K_est = self.net_kp(self.kernel_code).view(1, 1, self.kernel_size, self.kernel_size)
                        # out_y = fft2(x0_t_hat) * fft2(self.K_est, 256, 256)
                        # lossk = Complex_MSELoss(out_y, y_ft)
                        out_k = torch.nn.functional.conv2d(x0_t_hat.clone().detach(), self.K_est.expand(3, -1, -1, -1), groups=3,padding=self.kernel_size//2)
                        lossk = self.mse(out_k, y)
                        lossk.requires_grad = True
                        lossk.backward(retain_graph=True)
                        # lossk.detach()
                        self.optimizer_kp.step()
                        self.optimizer_kp.zero_grad()


                    # update kernel network via Monte Carlo simulations
                    if self.K_complex_flag == True:
                        self.K_est_fft = self.net_kp(self.kernel_code).view(1, 1, self.kernel_size, self.kernel_size)
                        kernel_random = self.obtain_kernel()
                        kernel_random_fft = fft2(kernel_random)
                        lossk = Complex_MSELoss(kernel_random_fft, self.K_est_fft)
                        lossk.requires_grad = True
                        lossk.backward(retain_graph=True)
                        lossk.detach()
                        self.optimizer_kp.step()
                        self.optimizer_kp.zero_grad()
                        self.K_est = ifft2(self.K_est_fft)

                    else:
                        for L in range(self.L):

                            self.obtain_DKP_kernel()
                            kernel_random = self.obtain_kernel()
                            lossk = self.KL_Loss(kernel_random, self.K_est)  # self.mse
                            lossk.requires_grad = True
                            lossk.backward(retain_graph=True)
                            lossk.detach()
                            self.optimizer_kp.step()
                            self.optimizer_kp.zero_grad()


                # image_psnr, image_ssim = evaluation_image(x0_t_hat, xgt, sf=4)
                # kernel_psnr = calculate_psnr(move2cpu(self.K_gt.squeeze()), move2cpu(self.K_est.squeeze()),
                #                              is_kernel=True)

                # print('Intermedia Evalution: PSNR:{:.2f}, SSIM:{:.4f}, Kernel PSNR:{}'.format(image_psnr, image_ssim, kernel_psnr))


                # Experiment3 large kernel size sf=2 & kernel_sizeX2
                # self.K_est = self.net_kp(self.kernel_code).view(1, 1, self.kernel_size, self.kernel_size)
                # kernel_random = self.gen_kernel_random(self.kernel_size, 4, 0)
                #
                # lossk = self.mse(kernel_random, self.K_est)
                # lossk.requires_grad = True
                # lossk.backward(retain_graph=True)
                # lossk.detach()
                # self.optimizer_kp.step()
                # self.optimizer_kp.zero_grad()


                # k_ft = fft2(self.K_est, 256, 256)
                #
                # k_ft.requires_grad = True
                # lossk = Complex_MSELoss(y_ft, x0_t_ft * k_ft)
                # lossk = Complex_MSELoss(y_ft, k_ft)


                # print("Kernel PSNR:{}".format(kernel_psnr))

                # mask-shift trick 
                # if model_kwargs['shift_w']==0 and model_kwargs['shift_h']==0:
                #     pass
                # elif model_kwargs['shift_w']==0 and model_kwargs['shift_h']!=0:#最左边那列
                #     h_l = int(128*model_kwargs['shift_h'])
                #     h_r = h_l+128
                #     if (model_kwargs['shift_h']==model_kwargs['shift_h_total']-1) and (model_kwargs['H_target']%128!=0):
                #         h_l = h_l-128+model_kwargs['H_target']%128
                #         x0_t_hat[:,:,0:256-model_kwargs['H_target']%128,:] = model_kwargs['x_temp'][:,:,h_l:h_r,0:256].to('cuda')
                #     else:
                #         x0_t_hat[:,:,0:128,:] = model_kwargs['x_temp'][:,:,h_l:h_r,0:256].to('cuda')
                # else:
                #     w_l = int(128*model_kwargs['shift_w'])
                #     w_r = w_l+128
                #     h_l = int(128*model_kwargs['shift_h'])
                #     h_r = h_l+256
                #     if (model_kwargs['shift_w']==model_kwargs['shift_w_total']-1) and (model_kwargs['W_target']%128!=0):
                #         w_l = w_l-128+model_kwargs['W_target']%128
                #         if (model_kwargs['shift_h']==model_kwargs['shift_h_total']-1) and (model_kwargs['H_target']%128!=0):
                #             h_l_tmp = h_l-128+model_kwargs['H_target']%128
                #             x0_t_hat[:,:,:,0:256-model_kwargs['W_target']%128] = model_kwargs['x_temp'][:,:,h_l_tmp:h_r,w_l:w_r].to('cuda')
                #         else:
                #             x0_t_hat[:,:,:,0:256-model_kwargs['W_target']%128] = model_kwargs['x_temp'][:,:,h_l:h_r,w_l:w_r].to('cuda')
                #     else:
                #         if (model_kwargs['shift_h']==model_kwargs['shift_h_total']-1) and (model_kwargs['H_target']%128!=0):
                #             h_l_tmp = h_l-128+model_kwargs['H_target']%128
                #             x0_t_hat[:,:,:,0:128] = model_kwargs['x_temp'][:,:,h_l_tmp:h_r,w_l:w_r].to('cuda')
                #         else:
                #             x0_t_hat[:,:,:,0:128] = model_kwargs['x_temp'][:,:,h_l:h_r,w_l:w_r].to('cuda')
                #     if model_kwargs['shift_h']!=0:
                #         h_r = h_l+128
                #         w_r = w_l+256
                #         if (model_kwargs['shift_h']==model_kwargs['shift_h_total']-1) and (model_kwargs['H_target']%128!=0):
                #             h_l = h_l-128+model_kwargs['H_target']%128
                #             x0_t_hat[:,:,0:256-model_kwargs['H_target']%128,:] = model_kwargs['x_temp'][:,:,h_l:h_r,w_l:w_r].to('cuda')
                #         else:
                #             x0_t_hat[:,:,0:128,:] = model_kwargs['x_temp'][:,:,h_l:h_r,w_l:w_r].to('cuda')

                # save intermediate results
                if t[0]%4==0:  # 25
                    # image_savepath = os.path.join('results/'+model_kwargs['save_path']+'/'+str(model_kwargs['shift_h'])+'_'+str(model_kwargs['shift_w']))
                    # os.makedirs(image_savepath, exist_ok=True)
                    # save_image(x0_t_hat[0], image_savepath, t[0])

                    image_psnr, image_ssim = evaluation_image(x0_t_hat, xgt, sf=4)
                    self.kernel_psnr = calculate_psnr(move2cpu(self.K_gt.squeeze()), move2cpu(self.K_est.squeeze()),
                                                 is_kernel=True)
                    print('Intermedia Evalution: PSNR:{:.2f}, SSIM:{:.4f}, Kernel PSNR:{}'.format(image_psnr, image_ssim, self.kernel_psnr))
                    h = model_kwargs['shift_h']
                    w_total = model_kwargs['shift_w_total']
                    w = model_kwargs['shift_w']
                    plt.imsave(model_kwargs['image_savepath'] + '{}_{}.png'.format(h*w_total+w,t[0]), tensor2im01(x0_t_hat), vmin=0,
                               vmax=1., dpi=1)

                    save_kernel(self.K_est, self.image_savepath + 'K_est_{}_{}.png'.format(h*w_total+w,t[0]))


                    row = int((101-t[0]).cpu().numpy())
                    self.sheet.write(row, 1, '%.2f' % (image_psnr), self.black_style)
                    self.sheet.write(row, 2, '%.4f' % image_ssim, self.black_style)
                    self.sheet.write(row, 3, '%.2f' % self.kernel_psnr, self.black_style)
                    self.wb.save(self.xlwt_save_path)

                model_mean, _, _ = self.q_posterior_mean_variance(x_start=x0_t_hat, x_t=x, t=t)
                model_variance = gamma_t # model_variance                
            
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == x0_t_hat.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "x0_t": x0_t_hat,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(
                self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """

        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)


        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] *
            gradient.float()
        )
        return new_mean



    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        conf=None,
        meas_fn=None,
        x0_t=None,
        idx_wall=-1
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'x0_t': a prediction of x_0.
        """
        

        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs
        )

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        ) 

        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )

        noise = th.randn_like(x)
        #sample = out["mean"] + nonzero_mask * \
        #    th.exp(0.5 * out["log_variance"]) * noise# - out["xt_grad"]
        
        sample = out["mean"] + nonzero_mask * \
            th.sqrt(th.ones(1,device='cuda')*out["variance"]) * noise

        result = {"sample": sample,
                  "x0_t": out["x0_t"], 'gt': model_kwargs.get('gt')}

        return result

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        return_all=False,
        conf=None
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        sample = self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            conf=conf
        )
        final = sample

        if return_all:
            return final
        else:
            return final["sample"]



    def deblur(self, Y_tensor, kernel_tensor):

        Y = Y_tensor.squeeze(0).cpu().numpy()
        Y = Y.transpose(1, 2, 0)

        kernel = kernel_tensor.squeeze(0).squeeze(0)
        kernel = kernel.cpu().numpy()
        kernel_ft = fftpack.fft2(kernel, shape=Y.shape[:2], axes=(0, 1))

        # convolve
        Y_ft = fftpack.fft2(Y, axes=(0, 1))
        # the 'newaxis' is to match to color direction
        X_ft = kernel_ft[:, :, np.newaxis] * Y_ft
        X = fftpack.ifft2(X_ft, axes=(0, 1)).real

        X = X.transpose(2, 0, 1)
        X = torch.from_numpy(X).unsqueeze(0)

        return X.to(self.device)

    def fft_blur(self, X, K):
        image_size1 = X.shape[-2]
        image_size2 = X.shape[-1]
        X_fft = fft2(X)
        K_fft256 = fft2(K, image_size1, image_size2)
        Y_fftm = X_fft * K_fft256
        Y = ifft2(Y_fftm, image_size1, image_size2)
        # Y = Move_X_forward(Y, move_x=9)
        return Y

    def fft_deblur(self, Y, K):
        image_size1 = Y.shape[-2]
        image_size2 = Y.shape[-1]
        Y_fft = fft2(Y)
        K_fft256 = fft2(K, image_size1, image_size2)
        X_fftm = Y_fft / (K_fft256)  # +1e-5
        X = ifft2(X_fftm, image_size1, image_size2)
        # X = Move_X_backward(X, move_x=9)
        return X



    def rotate(self, image, angle, center=None, scale=1.0):
        # grab the dimensions of the image
        (h, w) = image.shape[:2]

        # if the center is None, initialize it as the center of
        # the image
        if center is None:
            center = (w // 2, h // 2)

        # perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        # return the rotated image
        return rotated

    def gen_kernel_motion_fixed(self, k_size, sf, lens, theta, noise):

        # kernel_size = min(sf * 4 + 3, 21)
        kernel_size = k_size
        M = int((sf * 3 + 3) / 2)
        kernel_init = np.zeros([min(sf * 4 + 3, 21), min(sf * 4 + 3, 21)])
        # kernel_init[M-1:M+1,M-len:M-len] = 1
        kernel_init[M:M + 1, M - lens:M + lens + 1] = 1
        kernel = kernel_init + noise
        center = ((sf * 3 + 3) / 2, (sf * 3 + 3) / 2)
        kernel = self.rotate(kernel, theta, center, scale=1.0)

        kernel = kernel / np.sum(kernel)

        return kernel

    def gen_kernel_random_motion(self, noise_level=0):

        jj = 3
        motion_blur_path = 'D:\Codes\DDNM-master\DDNM-main-yzx-10-7\DDNM-main\hq_demo_hxh8-11\hq_demo\data\datasets\deblur\motion_kernel_j3_x4-8'
        num = len(os.listdir(motion_blur_path)) // 2
        random_num = int(np.random.rand() * num)

        kernel = sio.loadmat(os.path.join(motion_blur_path, "MotionKernel_{}_{}".format(random_num, jj)))['Kernel']

        kernel = torch.from_numpy(kernel).type(torch.FloatTensor).to(torch.device('cuda')).unsqueeze(0).unsqueeze(0)

        return kernel


    def gen_kernel_random(self, noise_level = 0):

        min_var = 0.175 * self.sf / 2
        max_var = min(2.5 * self.sf, 10) / 2 + self.width_add

        # move_x = (int(sf) + 1) * 3 / 2
        # move_x = ((sf * 5) - 2) / 2
        move_x = 7.5  # self.kernel_size / 2 - 1
        move_y = move_x

        lambda_1 = min_var + np.random.rand() * (max_var - min_var);
        lambda_2 = min_var + np.random.rand() * (max_var - min_var);
        theta = np.random.rand() * np.pi
        noise = -noise_level + np.random.rand(self.kernel_size) * noise_level * 2

        kernel = gen_kernel_fixed(self.kernel_size, self.sf, lambda_1, lambda_2, theta, noise, move_x, move_y)

        kernel = torch.from_numpy(kernel).type(torch.FloatTensor).to(torch.device('cuda')).unsqueeze(0).unsqueeze(0)

        return kernel

    def gen_kernel_motion_line_fixed(self, k_size, sf, lens, theta, noise):

        # kernel_size = min(sf * 4 + 3, 21)
        kernel_size = k_size
        M = kernel_size // 2
        # M = int((sf[0]*3+3)/2)
        kernel_init = np.zeros([k_size, k_size])
        # kernel_init[M-1:M+1,M-len:M-len] = 1
        kernel_init[M, M - lens:M + lens + 1] = 1
        kernel = kernel_init + noise
        # center = ((sf[0]*3+3)/2, (sf[0]*3+3)/2)  # left-top
        center = (k_size// 2, k_size// 2)
        kernel = self.rotate(kernel, theta, center, scale=1.0)

        kernel = kernel / np.sum(kernel)

        return kernel

    def gen_kernel_line_motion(self, lens = 3, noise_level = 0):
        # lambda_1 = min_var + np.random.rand() * (max_var - min_var);
        # lambda_2 = min_var + np.random.rand() * (max_var - min_var);
        # theta = np.random.rand() * 360 # np.pi

        theta = np.random.randint(int(180/self.random_angle)) * self.random_angle

        noise = -noise_level + np.random.rand(self.kernel_size) * noise_level * 2

        kernel = self.gen_kernel_motion_line_fixed(self.kernel_size, self.sf, lens, theta, noise)

        kernel = torch.from_numpy(kernel).type(torch.FloatTensor).to(torch.device('cuda')).unsqueeze(0).unsqueeze(0)

        return kernel

    def kernel_4_read(self, noise_level = 0):

        path = 'D:\Codes\DDNM-master\DDNM-main-yzx-10-7\DDNM-main\hq_demo_hxh8-11\hq_demo\data\datasets\deblur\kernels_12.mat'
        kernels = sio.loadmat(path)

        k_num = np.random.randint(4) + 8

        kernel = kernels['kernels'][0, k_num]

        kernel = torch.from_numpy(kernel).type(torch.FloatTensor).to(torch.device('cuda')).unsqueeze(0).unsqueeze(0)

        incorp_mode_k = 'bicubic'  # nearest, bicubic
        kernel = torch.nn.functional.interpolate(kernel,
                                                       size=[self.kernel_size, self.kernel_size],
                                                       mode=incorp_mode_k)  # 卷积完模糊核后下采样

        aa = sum(sum(sum(sum(kernel))))
        kernel = kernel / aa

        return kernel

    def kernel_read_text(self, path):

        kernel = sio.loadmat(path)['Kernel']

        kernel = torch.from_numpy(kernel).type(torch.FloatTensor).to(torch.device('cuda')).unsqueeze(0).unsqueeze(0)

        incorp_mode_k = 'bicubic'  # nearest, bicubic
        kernel = torch.nn.functional.interpolate(kernel,
                                                       size=[self.kernel_size, self.kernel_size],
                                                       mode=incorp_mode_k)  # 卷积完模糊核后下采样

        aa = sum(sum(sum(sum(kernel))))
        kernel = kernel / aa

        return kernel


    def obtain_kernel(self):

        if self.kernel_type == 'Gaussian':
            return self.gen_kernel_random()
        elif self.kernel_type == 'Motion_line':
            return self.gen_kernel_line_motion()
        elif self.kernel_type == 'Motion':
            return self.gen_kernel_random_motion()
        elif self.kernel_type == 'Motion_4':
            return self.kernel_4_read()


    def obtain_DKP_kernel(self):

        self.K_est = self.net_kp(self.kernel_code).view(1, 1, self.kernel_size, self.kernel_size)



    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        conf=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            image_after_step = noise
        else:
            image_after_step = th.randn(*shape, device=device)

        x0_t = None
        self.device = device

        # initialization
        gt = model_kwargs['gt'] 
        scale = model_kwargs['scale'] 

        if 256%scale!=0:
            raise ValueError("Please set a SR scale divisible by 256")
        if gt.shape[2]!=256 and conf.name=='face256':
            print("gt.shape:",gt.shape)
            raise ValueError("Only support output size 256x256 for face images")#面部表情只支持256*265的输入

        if model_kwargs['resize_y']:
            resize_y = lambda z: MeanUpsample(z,scale)
            gt = resize_y(gt)#先对gt进行上采样，以上采样后的gt为x_org进行和diffusion model一样的操作


        if model_kwargs['deg']=='sr_averagepooling':
            scale=model_kwargs['scale'] 
            A = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
            Ap = lambda z: MeanUpsample(z,scale)

            A_temp = torch.nn.AdaptiveAvgPool2d((gt.shape[2]//scale,gt.shape[3]//scale))

        elif model_kwargs['deg'] == 'super-resolution':

            A = lambda X_pad, kernel_tensor: F.conv2d(X_pad, kernel_tensor.expand(3, -1, -1, -1).clone().detach(),
                                                      groups=3)

            Ap = lambda Y, kernel_tensor: self.deblur(Y, kernel_tensor)

            A_temp = A

        elif model_kwargs['deg']=='deblur':
            image_savepath = os.path.join(
                'results/' + model_kwargs['save_path'] + '/{}/'.format(model_kwargs['image_num']))
            model_kwargs['image_savepath'] = image_savepath
            self.image_savepath = image_savepath
            os.makedirs(image_savepath, exist_ok=True)
            self.xlwt_save_path = self.image_savepath + '/intermediate_results.xlsx'

            self.wb = xlwt.Workbook()  # 一个实例
            self.sheet = self.wb.add_sheet("Sheet1")  # 工作簿名称
            style = "font:colour_index black;"  # 设置样式
            self.black_style = xlwt.easyxf(style)
            self.sheet.write(0, 1, "image PSNR")
            self.sheet.write(0, 2, "image SSIM")
            self.sheet.write(0, 3, "kernel PSNR")

            for i in range(1, 100):
                self.sheet.write(i, 0, i)  # 不设置格式
                self.wb.save(self.xlwt_save_path)  # 最后一定要保存，否则无效


            # A = lambda X_pad, kernel_tensor: F.conv2d(X_pad, kernel_tensor.expand(3, -1, -1, -1).clone().detach(), groups=3)
            # Ap = lambda Y, kernel_tensor: self.deblur(Y, kernel_tensor)
            A = lambda X, kernel_tensor: self.fft_blur(X, kernel_tensor)
            Ap = lambda Y, kernel_tensor: self.fft_deblur(Y, kernel_tensor)
            A_temp = A
            self.sf = model_kwargs['scale']
            self.kernel_size = 19  #  sf * 4 + 3
            self.alph_beta = 0.1 # 0.1 FOR Gaussian and motion 越大diffusion比重越大
            self.kernel_type = model_kwargs['kernel_type'] # 'Gaussian' 'Motion_line' 'Motion' , 'Motion_4'
            self.Ki = 1
            self.L = 1  # 越大越倾向于均匀高斯
            self.XK = 1  # 大学习率往里缩，小学习率影响不大

            self.time_back = 1  # visual for 3

            #  60 4 1 for Gaussian,  1 1 1 for motion
            self.random_angle = 30
            self.k_hiddenlayer = 1000
            self.width_add = 0.5 # 0.5 for Gaussian
            # self.K_gt =  self.kernel_read_text(model_kwargs['k_mat_path'])
            self.K_gt = self.obtain_kernel()

            self.width_add = 2 # 2 for Gaussian
            self.ssimloss = SSIM().to(device)
            self.mse = torch.nn.MSELoss().to(device)
            self.KL_Loss = KL_Loss

            self.K_complex_flag = False

            n_k = 200
            if self.K_complex_flag == True:
                self.kernel_code = get_noise(n_k, 'noise', (1, 1)).detach().squeeze().to(device).type(torch.complex64)
                self.kernel_code.requires_grad = False
                self.net_kp = ComplexFCN(n_k, self.kernel_size ** 2).to(device)
            else:
                self.kernel_code = get_noise(n_k, 'noise', (1, 1)).detach().squeeze().to(device)
                self.kernel_code.requires_grad = False
                self.net_kp = fcn(n_k, self.kernel_size ** 2, self.k_hiddenlayer).to(device)


            self.optimizer_kp = torch.optim.Adam([{'params': self.net_kp.parameters()}], lr=model_kwargs['kp_lr'])  # 1e-4
            log_str = 'Number of parameters in Generator-k: {:.2f}K'
            print(log_str.format(calculate_parameters(self.net_kp) / 1000))

            self.obtain_DKP_kernel()

            for i in range(self.Ki):   #200

                # Experiment3 large kernel size sf=2 & kernel_sizeX2
                if self.K_complex_flag == True:
                    self.K_est_fft = self.net_kp(self.kernel_code).view(1, 1, self.kernel_size, self.kernel_size)
                    kernel_random = self.obtain_kernel()
                    kernel_random_fft = fft2(kernel_random)
                    lossk = Complex_MSELoss(kernel_random_fft, self.K_est_fft)
                    lossk.backward(retain_graph=True)
                    lossk.detach()
                    self.optimizer_kp.step()
                    self.optimizer_kp.zero_grad()

                    self.K_est = ifft2(self.K_est_fft)

                else:
                    self.obtain_DKP_kernel()
                    kernel_random = self.K_gt # self.obtain_kernel()  #
                    # kernel_random = self.gen_kernel_random()  #  self.K_gt
                    lossk = self.KL_Loss(kernel_random, self.K_est)
                    lossk.backward(retain_graph=True)
                    lossk.detach()
                    self.optimizer_kp.step()
                    self.optimizer_kp.zero_grad()


            model_kwargs['kernel_tensor'] = self.K_est


        elif model_kwargs['deg']=='inpainting' and conf.name=='face256':
        # elif model_kwargs['deg'] == 'inpainting':
            mask = model_kwargs.get('gt_keep_mask')
            A = lambda z: z*mask
            Ap = A

            A_temp = A
        elif model_kwargs['deg']=='mask_color_sr' and conf.name=='face256':
            mask = model_kwargs.get('gt_keep_mask')
            A1 = lambda z: z*mask
            A1p = A1
            
            A2 = lambda z: color2gray(z)
            A2p = lambda z: gray2color(z)
            
            scale=model_kwargs['scale']
            A3 = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
            A3p = lambda z: MeanUpsample(z,scale)
            
            A = lambda z: A3(A2(A1(z)))
            Ap = lambda z: A1p(A2p(A3p(z)))

            A_temp = A    
        elif model_kwargs['deg']=='colorization':
            A = lambda z: color2gray(z)
            Ap = lambda z: gray2color(z)

            A_temp = A
        elif model_kwargs['deg']=='sr_color':
            scale=model_kwargs['scale'] 
            A1 = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
            A1p = lambda z: MeanUpsample(z,scale)
            A2 = lambda z: color2gray(z)
            A2p = lambda z: gray2color(z)
            A = lambda z: A2(A1(z))
            Ap = lambda z: A1p(A2p(z))
            A1_temp = torch.nn.AdaptiveAvgPool2d((gt.shape[2]//scale,gt.shape[3]//scale))
            A_temp = lambda z: A2(A1_temp(z))            
        else:
            raise NotImplementedError("degradation type not supported")         

        model_kwargs['A'] = A
        model_kwargs['Ap'] = Ap

        if model_kwargs['deg']=='deblur':

            self.Downsample = torch.nn.AdaptiveAvgPool2d((256 // scale, 256 // scale))
            self.Upsample = lambda z: MeanUpsample(z, scale)

            if model_kwargs['dataset_name'] != 'CVPR-real':

                # gt_pad = F.pad(gt, mode='replicate', pad=(k_size // 2, k_size // 2, k_size // 2, k_size // 2))
                # y_temp = A_temp(gt_pad, kernel_tensor)
                # Apy_temp = Ap(y_temp, kernel_tensor)
                X = gt.to(device)

                plt.imsave(image_savepath + 'GT.png', tensor2im01(X), vmin=0,
                           vmax=1., dpi=1)

                X_fft = fft2(X)
                plt.imsave(os.path.join(image_savepath, 'X_fft.png'), tensor2im01(X_fft), vmin=0,
                           vmax=1., dpi=1)

                # output_path = image_savepath

                #
                # X_fft_ifft = ifft2(X_fft)
                # plt.imsave(os.path.join(image_savepath, 'X_fft_ifft.png'), tensor2im01(X_fft_ifft), vmin=0,
                #            vmax=1., dpi=1)
                # #

                K_fft = fft2(self.K_gt)
                save_kernel(K_fft, image_savepath + 'K_fft')

                K_est_fft = fft2(self.K_est)
                save_kernel(K_est_fft, image_savepath + 'K_est_fft')


                #
                # K_fft_ifft = ifft2(K_fft)
                # save_kernel(K_fft_ifft, output_path + 'K_fft_ifft')
                #
                # K_fft256 = fft2(K, size=256)
                # save_kernel(K_fft256, output_path + 'K_fft256')
                #
                # K_fft256_ifft19 = ifft2(K_fft256, size=19)
                # save_kernel(K_fft256_ifft19, output_path + 'K_fft256_ifft19')
                #

                # plt.imsave(os.path.join(output_path, 'Y.png'), tensor2im01(Y), vmin=0,
                #            vmax=1., dpi=1)

                #
                # Y_fft_ifft = ifft2(Y_fft)
                # plt.imsave(os.path.join(output_path, 'Y_fft_ifft.png'), tensor2im01(Y_fft_ifft), vmin=0,
                #            vmax=1., dpi=1)
                #
                # Y_fftm = X_fft * K_fft256
                # plt.imsave(os.path.join(output_path, 'Y_fftm.png'), tensor2im01(Y_fftm), vmin=0,
                #            vmax=1., dpi=1)
                #
                # Y_fftm_ifftm = ifft2(Y_fftm)
                # plt.imsave(os.path.join(output_path, 'Y_fftm_ifftm.png'), tensor2im01(Y_fftm_ifftm), vmin=0,
                #            vmax=1., dpi=1)
                #
                # X_fftp = Y_fftm / K_fft256
                # plt.imsave(os.path.join(output_path, 'X_fftp.png'), tensor2im01(X_fftp), vmin=0,
                #            vmax=1., dpi=1)
                #
                # X_fftp_ifftp = ifft2(X_fftp)
                # plt.imsave(os.path.join(output_path, 'X_fftp_ifftp.png'), tensor2im01(X_fftp_ifftp), vmin=0,
                #            vmax=1., dpi=1)
                #
                # delta_X = X_fftp - X_fft
                # plt.imsave(os.path.join(output_path, 'delta_X.png'), tensor2im01(delta_X), vmin=0,
                #            vmax=1., dpi=1)

                self.X_gt = X
                save_kernel(self.K_gt, image_savepath + 'K_gt.png')
                save_kernel(self.K_est, image_savepath + 'K_est_initial.png')
                self.kernel_psnr = calculate_psnr(move2cpu(self.K_gt.squeeze()), move2cpu(self.K_est.squeeze()), is_kernel=True)

                # #将卷积运算修改为相关运算
                # self.K_gt_corr = cv2.flip(tensor2im01(self.K_gt), -1)#先转为numpy，再沿x轴和y轴各翻转一次
                # self.K_gt_corr = np.expand_dims(self.K_gt_corr, axis = 2)#添加一个维度，变为3维
                # self.K_gt_corr = torch.unsqueeze(im2tensor01(self.K_gt_corr), dim = 0)#转换回tensor，添加一个维度
                # save_kernel(self.K_gt_corr, image_savepath + 'K_gt_corr.png')


                Y_unsf_unmoved = A(X, self.K_gt)#图像真值X与高斯核真值K_gt的卷积
                vutils.save_image(Y_unsf_unmoved, image_savepath +'Y_unsf_unmoved.png', normalize=True)

                Y_unsf = Move_X_forward(Y_unsf_unmoved)
                #vutils.save_image(Y_unsf, image_savepath + 'Y_unsf.png', normalize=True)

                sf_bic = 3

                incorp_mode = 'bicubic'   # nearest, bicubic
                self.Y_small = torch.nn.functional.interpolate(abs(Y_unsf), size=[Y_unsf.shape[-2] // sf_bic, Y_unsf.shape[-1] // sf_bic], mode=incorp_mode)  #卷积完模糊核后下采样
                Y = torch.nn.functional.interpolate(self.Y_small, size=[Y_unsf.shape[-2], Y_unsf.shape[-1]], mode=incorp_mode) # , align_corners=False  逆变换，先上采样


                self.Apy_temp = Ap(Y, self.K_est)  # self.K_est 逆FFT


                # self.Apy_temp = Move_X_backward(self.Apy_temp, move_x=9)

                plt.imsave(image_savepath + 'Y.png', tensor2im01(Y), vmin=0,
                           vmax=1., dpi=1)
                plt.imsave(image_savepath + 'Apy.png', tensor2im01(self.Apy_temp), vmin=0,
                           vmax=1., dpi=1)
                Y_fft = fft2(Y)
                plt.imsave(os.path.join(image_savepath, 'Y_fft.png'), tensor2im01(Y_fft), vmin=0,
                           vmax=1., dpi=1)

                Y_fft_ifft = ifft2(Y_fft)
                plt.imsave(os.path.join(image_savepath, 'Y_fft_ifft.png'), tensor2im01(Y_fft_ifft), vmin=0,
                           vmax=1., dpi=1)
                Apy_fft = fft2(self.Apy_temp)
                plt.imsave(os.path.join(image_savepath, 'Apy_fft.png'), tensor2im01(Apy_fft), vmin=0,
                           vmax=1., dpi=1)


                image_psnr, image_ssim = evaluation_image(self.Apy_temp, self.X_gt, sf=4)
                print('Apy evalution: PSNR:{:.2f}, SSIM:{:.4f}, kernel PSNR:{:.2f}'.format(image_psnr, image_ssim, self.kernel_psnr))

                # plt.imsave(os.path.join(image_savepath, 'Apy_temp.png'), tensor2im01(self.Apy_temp), vmin=0,
                #            vmax=1., dpi=1)

                y_temp = Y
                model_kwargs['y_temp'] = y_temp

            elif model_kwargs['dataset_name'] == 'CVPR-real':

                Y = gt.to(device)
                plt.imsave(image_savepath + 'Y.png', tensor2im01(Y), vmin=0,
                           vmax=1., dpi=1)

                incorp_mode = 'bicubic'  # nearest, bicubic

                Y = torch.nn.functional.interpolate(Y, size=[256*2, 256*2],
                                                    mode=incorp_mode)  # , align_corners=False  逆变换，先上采样

                self.X_gt = Y
                Y_unsf = Y
                self.Apy_temp = Ap(Y, self.K_est)
                y_temp = Y
                model_kwargs['y_temp'] = y_temp

            # _, C, H, W = gt.size()
            #
            # self.net_dip =  tiny_skip(num_input_channels=C,
            #                      num_output_channels=C,
            #                      num_channels_down=[96, 96, 96],
            #                      num_channels_up=[96, 96, 96],
            #                      num_channels_skip=16,
            #                      upsample_mode='bilinear',
            #                      need_sigmoid=True,
            #                      need_bias=True,
            #                      pad='reflection',
            #                      act_fun='LeakyReLU',
            #                      use_bn=True).cuda()
            #
            # self.net_dip = self.net_dip.to(device)
            # self.optimizer_dip = torch.optim.Adam([{'params': self.net_dip.parameters()}], lr=model_kwargs['dip_lr'])
            # self.ssimloss = SSIM().to(device)
            # self.mse = torch.nn.MSELoss().to(device)

            print('*' * 60 + '\nSTARTED DDNM on Dre:{}, data:{}'.format(model_kwargs['deg'], model_kwargs['path_y']))

            # dip_sr = self.net_dip(y_temp)


        else:
            y_temp = A_temp(gt)
            self.Apy_temp = Ap(y_temp)

        # y_savepath = os.path.join('results/' + 'tuning' + '/y')
        # os.makedirs(y_savepath, exist_ok=True)
        # save_image(y_temp.squeeze(), y_savepath, 0)
        #
        # Apy_savepath = os.path.join('results/' + 'tuning' + '/Apy')
        # os.makedirs(Apy_savepath, exist_ok=True)
        # save_image(Apy_temp.squeeze(), Apy_savepath, 0)

        Apy_temp = self.Apy_temp


        #假设256*256的原图gt要超分2倍，变成512*512,实现方法是gt先超分2倍
        #然后卷积模糊核K，再下采样2倍得到y_temp：256*256
        #y_temp上采样2倍，再做反卷积，得到Apy_temp：512*512
        H_target, W_target = Apy_temp.shape[2], Apy_temp.shape[3]
        model_kwargs['H_target'] = H_target
        model_kwargs['W_target'] = W_target

        if (H_target<256 or W_target<256) and model_kwargs['dataset_name'] != 'CVPR-real':
            raise ValueError("Please set a larger SR scale")

        # image_savepath = os.path.join('results/'+model_kwargs['save_path']+'/Apy')
        # os.makedirs(image_savepath, exist_ok=True)
        # save_image(Apy_temp[0], image_savepath, 0)
        #
        # image_savepath = os.path.join('results/'+model_kwargs['save_path']+'/y')
        # os.makedirs(image_savepath, exist_ok=True)
        # save_image(y_temp[0], image_savepath, 0)
        #
        # image_savepath = os.path.join('results/'+model_kwargs['save_path']+'/gt')
        # os.makedirs(image_savepath, exist_ok=True)
        # save_image(gt[0], image_savepath, 0)

        finalresult = torch.zeros_like(Apy_temp)

        shift_h_total = math.ceil(H_target/128)-1 #向下取整，1.01变为1，2变为1，2.01变为2
        shift_w_total = math.ceil(W_target/128)-1
        model_kwargs['shift_h_total'] = shift_h_total
        model_kwargs['shift_w_total'] = shift_w_total

        with tqdm(total=shift_h_total*shift_w_total) as pbar:
            pbar.set_description('total shifts')

            # shift along H
            for shift_h in range(shift_h_total):
                h_l = int(128*shift_h)
                h_r = h_l+256
                if (shift_h==shift_h_total-1) and (H_target%128!=0): # for the last irregular shift_h
                    h_r = Apy_temp.shape[2]
                    h_l = h_r-256

                # shift along W
                for shift_w in range(shift_w_total):

                    x_temp=finalresult
                    w_l = int(128*shift_w)
                    w_r = w_l+256
                    if (shift_w==shift_w_total-1) and (W_target%128!=0): # for the last irregular shift_w
                        w_r = Apy_temp.shape[3]
                        w_l = w_r-256

                    # get the shifted y
                    Apy_patch = Apy_temp[:,:,h_l:h_r,w_l:w_r]#退化图Y上采样后再做反卷积512*512
                    y_patch = y_temp[:,:,h_l:h_r,w_l:w_r]#退化图Y 256*256
                    Xgt_patch = self.X_gt[:,:,h_l:h_r,w_l:w_r]#resize参数后的gt，512*512
                    Y_small_patch = Y_unsf[:,:,h_l:h_r,w_l:w_r]#仅将gt和k卷积后得到的Y


                    self.Y_small_patch = torch.nn.functional.interpolate(Y_small_patch, size=[Y_small_patch.shape[-2] // self.sf,
                                                                                      Y_small_patch.shape[-1] // self.sf],
                                                                   mode=incorp_mode)  #802行

                    model_kwargs['shift_w'] = shift_w
                    model_kwargs['shift_h'] = shift_h
                    model_kwargs['y_patch'] = y_patch
                    model_kwargs['Apy_patch'] = Apy_patch
                    model_kwargs['Xgt_patch'] = Xgt_patch
                    model_kwargs['x_temp'] = x_temp

                    times = get_schedule_jump(**conf.schedule_jump_params)

                    time_pairs = list(zip(times[:-self.time_back], times[self.time_back:]))

                    # DDNM loop
                    for t_last, t_cur in tqdm(time_pairs):
                        t_last_t = th.tensor([t_last] * shape[0],
                                             device=device)

                        # normal DDNM sampling
                        if t_cur < t_last:  
                            with th.no_grad():
                                image_before_step = image_after_step.clone()

                                tt = torch.ones(1) * 98

                                # flops, params = profile(model=model,
                                #                         inputs=(torch.randn(1, 3, 256, 256), tt.to(device),))

                                out = self.p_sample(
                                    model,
                                    image_after_step,
                                    t_last_t,
                                    clip_denoised=clip_denoised,
                                    denoised_fn=denoised_fn,
                                    cond_fn=cond_fn,
                                    model_kwargs=model_kwargs,
                                    conf=conf,
                                    x0_t=x0_t
                                )
                                image_after_step = out["sample"]
                                x0_t = out["x0_t"]

                        # time-travel back
                        else:
                            t_shift = conf.get('inpa_inj_time_shift', 1)

                            image_before_step = image_after_step.clone()
                            image_after_step = self.undo(
                                image_before_step, image_after_step,
                                est_x_0=out['x0_t'], t=t_last_t+t_shift, debug=False)
                            x0_t = out["x0_t"]




                    # save the shifted result
                    bias = 10


                    if (shift_w==shift_w_total-1) and (W_target%128!=0):#如果width最后一次，且除不尽，要重新规定宽度
                        if (shift_h==shift_h_total-1) and (H_target%128!=0):#如果height恰好最后一次，且除不尽，双重特殊情况
                            finalresult[:,:,int(128*shift_h+bias)-128+H_target%128:int(128*shift_h)+128+H_target%128,int(128*shift_w+bias)-128+W_target%128:int(128*shift_w)+128+W_target%128] = out["x0_t"][:,:,bias:,bias:]
                        elif(shift_h==0):
                            finalresult[:,:,int(128*shift_h):int(128*shift_h)+256,int(128*shift_w+bias)-128+W_target%128:int(128*shift_w)+128+W_target%128] = out["x0_t"][:,:,:,bias:]
                        else:
                            finalresult[:, :, int(128 * shift_h+bias):int(128 * shift_h) + 256,
                            int(128 * shift_w + bias) - 128 + W_target % 128:int(
                                128 * shift_w) + 128 + W_target % 128] = out["x0_t"][:, :, bias:, bias:]
                    else:
                        if (shift_h==shift_h_total-1) and (H_target%128!=0):#如果height恰好最后一次，且除不尽，但是width此时除的尽
                            if(shift_w==0):
                                finalresult[:,:,int(128*shift_h+bias)-128+H_target%128:int(128*shift_h)+128+H_target%128,int(128*shift_w):int(128*shift_w)+256] = out["x0_t"][:,:,bias:,:]
                            else:
                                finalresult[:, :, int(128 * shift_h + bias) - 128 + H_target % 128:int(
                                    128 * shift_h) + 128 + H_target % 128,
                                int(128 * shift_w+bias):int(128 * shift_w) + 256] = out["x0_t"][:, :, bias:, bias:]
                        else:
                            finalresult = shift_cut(finalresult, out["x0_t"], shift_h, shift_w, bias)

                    pbar.update(1)

        # finish!
        # image_savepath = os.path.join('results/'+model_kwargs['save_path']+'/final')
        # os.makedirs(image_savepath, exist_ok=True)

        out["kernel_psnr"] = self.kernel_psnr

        finalresult_fft = fft2(finalresult)
        plt.imsave(os.path.join(image_savepath, 'finalresult_fft.png'), tensor2im01(finalresult_fft), vmin=0,
                   vmax=1., dpi=1)

        plt.imsave(image_savepath + 'final.png', tensor2im01(finalresult), vmin=0,
                   vmax=1., dpi=1)

        # save_image(finalresult[0], image_savepath, 0)

        out["sample"] = finalresult
        return out

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
