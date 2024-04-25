
import os
import argparse
import torch

import torch.nn.functional as F
import time

import torchvision.transforms as transforms
from PIL import Image
import tqdm
import torch.utils.data as data
import yaml
import numpy as np
from scipy.ndimage import measurements, interpolation
import matplotlib.pyplot as plt
# Workaround
try:
    import ctypes

    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass

def move2cpu(d):
    """Move data from gpu to cpu"""
    return d.detach().cpu().float().numpy()


def im2tensor01(im_np):
    """Convert numpy to tensor to the gpu"""
    im_np = im_np / 255.0 if im_np.dtype == 'uint8' else im_np
    return torch.FloatTensor(np.transpose(im_np, (2, 0, 1))).cuda()

def tensor2im01(im_t):
    """Copy the tensor to the cpu & convert to range [0,255]"""
    im_np = np.clip(np.round((np.transpose(move2cpu(im_t).squeeze(0), (1, 2, 0))) * 255.0), 0, 255)
    return im_np.astype(np.uint8)

def read_image(path):
    """Loads an image"""
    im = Image.open(path).convert('RGB')
    im = np.array(im, dtype=np.uint8)
    return im

def tensor2im(var):
    # var shape: (3, H, W)
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def save_image(img, save_dir, idx):
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, f"{idx:05d}.png")
    Image.fromarray(np.array(result)).save(im_save_path)


def kernel_move(kernel, move_x, move_y):
    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)  # 寻找中心值

    current_center_of_mass_list = list(current_center_of_mass)
    shift_vec_list = list(current_center_of_mass)

    # Define the shift vector for the kernel shifting (x,y)
    shift_vec_list[0] = move_x - current_center_of_mass_list[0]
    shift_vec_list[1] = move_y - current_center_of_mass_list[1]

    shift_vec = tuple(shift_vec_list)

    # Finally shift the kernel and return
    return interpolation.shift(kernel, shift_vec)  # kernel的平移

def gen_kernel_fixed(k_s, scale_factor, lambda_1, lambda_2, theta, noise, move_x, move_y):
    k_size = (np.ones(2, dtype=int) * int(k_s))

    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2]);
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2 + 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # shift the kernel so it will be centered
    raw_kernel_moved = kernel_move(raw_kernel, move_x, move_y)

    # Normalize the kernel and return
    kernel = raw_kernel_moved / np.sum(raw_kernel_moved)
    # kernel = raw_kernel_centered / np.sum(raw_kernel_centered)

    return kernel

def gen_kernel_random(k_size, sf, noise_level):

    min_var = 0.175 * sf
    max_var = min(2.5 * sf, 10)

    # move_x = 0

    # move_x = (int(sf) + 1) * 3 / 2
    #
    move_x = ((int(sf) * 5) - 1) / 2  # center
    move_y = move_x

    lambda_1 = min_var + np.random.rand() * (max_var - min_var);
    lambda_2 = min_var + np.random.rand() * (max_var - min_var);
    theta = np.random.rand() * np.pi
    noise = -noise_level + np.random.rand(k_size) * noise_level * 2

    kernel = gen_kernel_fixed(k_size, sf, lambda_1, lambda_2, theta, noise, move_x, move_y)

    return kernel

def save_kernel_png(k, output_dir_path, name):
    """saves the final kernel and the analytic kernel to the results folder"""
    os.makedirs(os.path.join(output_dir_path), exist_ok=True)

    savepath_png = os.path.join(output_dir_path, name)

    k = move2cpu(k.squeeze())

    # sio.savemat(savepath_mat, {'Kernel': k})
    plot_kernel(k, k, savepath_png)

def plot_kernel(gt_k_np, out_k_np, savepath):
    plt.clf()
    f, ax = plt.subplots(1, 2, figsize=(6, 4), squeeze=False)
    im = ax[0, 0].imshow(gt_k_np, vmin=0, vmax=gt_k_np.max())
    plt.colorbar(im, ax=ax[0, 0])
    im = ax[0, 1].imshow(out_k_np, vmin=0, vmax=out_k_np.max())
    plt.colorbar(im, ax=ax[0, 1])
    ax[0, 0].set_title('GT')
    ax[0, 1].set_title('PSNR')

    plt.savefig(savepath)