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

import yaml
import os
from PIL import Image
import torchvision.transforms as transforms
import torch as th
import torch
import numpy as np
from scipy.ndimage import measurements, interpolation
import torchvision
from functools import partial
from torch.utils.data import Subset
import torchvision.transforms.functional as F
from .celeba import CelebA
from .lsun import LSUN
import torch.fft
import matplotlib.pyplot as plt
import cv2
import math
import torch.nn as nn


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

def center_crop_arr(pil_image, image_size = 256):
    # Imported from openai/guided-diffusion
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]



def txtread(path):
    path = os.path.expanduser(path)
    with open(path, 'r') as f:
        return f.read()


def yamlread(path):
    return yaml.safe_load(txtread(path=path))

def imwrite(path=None, img=None):
    Image.fromarray(img).save(path)


def get_dataset(args, config):
    if config.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.image_size), transforms.ToTensor()]
        )
    else:
        # 训练集进行数据增强，测试集不进行
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.image_size),  # 这里就把数据处理成256*256的了
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.image_size), transforms.ToTensor()]
        )

    if config.dataset == "CELEBA":
        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64
        if config.random_flip:
            dataset = CelebA(
                root=os.path.join(args.exp, "datasets", "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )
        else:
            dataset = CelebA(
                root=os.path.join(args.exp, "datasets", "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config.image_size),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )

        test_dataset = CelebA(
            root=os.path.join(args.exp, "datasets", "celeba"),
            split="test",
            transform=transforms.Compose(
                [
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(config.image_size),
                    transforms.ToTensor(),
                ]
            ),
            download=True,
        )

    elif config.dataset == "LSUN":
        if config.data.out_of_dist:
            dataset = torchvision.datasets.ImageFolder(
                os.path.join(args.exp, 'datasets', "ood_{}".format(config.data.category)),
                transform=transforms.Compose([partial(center_crop_arr, image_size=config.data.image_size),
                                              transforms.ToTensor()])
            )
            test_dataset = dataset
        else:
            train_folder = "{}_train".format(config.data.category)
            val_folder = "{}_val".format(config.data.category)
            test_dataset = LSUN(
                root=os.path.join(args.exp, "datasets", "lsun"),
                classes=[val_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.CenterCrop(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                )
            )
            dataset = test_dataset

    elif config.dataset == "CelebA_HQ" or config.dataset == 'FFHQ':
        if config.out_of_dist:
            dataset = torchvision.datasets.ImageFolder(
                os.path.join(args.exp, "datasets", "ood_celeba"),
                transform=transforms.Compose([transforms.Resize([config.image_size, config.image_size]),
                                              transforms.ToTensor()])
            )
            test_dataset = dataset
        else:
            dataset = torchvision.datasets.ImageFolder(
                os.path.join(args['exp'], "data/datasets", args['path_y']),  # os.path.join(args.exp, "datasets", "celeba_hq"),
                # transform=transforms.Compose([transforms.Resize([config.data.image_size, config.data.image_size]),
                #                 #                               transforms.ToTensor()])
                # transform=transforms.Compose([transforms.RandomCrop([config.image_size, config.image_size]),
                #                               transforms.Resize([config.image_size, config.image_size]),
                #                               transforms.ToTensor()])
                transform = transforms.Compose([transforms.Resize([config.image_size, config.image_size]),
                                                transforms.ToTensor()])
            )
            num_items = len(dataset)
            indices = list(range(num_items))
            random_state = np.random.get_state()
            np.random.seed(2019)
            np.random.shuffle(indices)
            np.random.set_state(random_state)
            #             train_indices, test_indices = (
            #                 indices[: int(num_items * 0.9)],
            #                 indices[int(num_items * 0.9) :],
            #             )
            train_indices, test_indices = (
                indices[: int(num_items * 0.)],
                indices[int(num_items * 0.):],
            )
            test_dataset = Subset(dataset, test_indices)

    elif config.dataset == 'ImageNet':
        # only use validation dataset here

        if config.data.subset_1k:
            from datasets.imagenet_subset import ImageDataset
            dataset = ImageDataset(os.path.join(args.exp, 'datasets', 'imagenet', 'imagenet'),
                                   os.path.join(args.exp, 'imagenet_val_1k.txt'),
                                   image_size=config.data.image_size,
                                   normalize=False)
            test_dataset = dataset
        elif config.data.out_of_dist:
            dataset = torchvision.datasets.ImageFolder(
                os.path.join(args.exp, 'datasets', 'ood'),
                transform=transforms.Compose([partial(center_crop_arr, image_size=config.data.image_size),
                                              transforms.ToTensor()])
            )
            test_dataset = dataset
        else:
            dataset = torchvision.datasets.ImageNet(
                os.path.join(args.exp, 'datasets', 'imagenet'), split='val',
                transform=transforms.Compose([partial(center_crop_arr, image_size=config.data.image_size),
                                              transforms.ToTensor()])
            )
            test_dataset = dataset
    else:
        dataset, test_dataset = None, None

    return dataset, test_dataset


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


def color2gray(x):
    coef = 1 / 3
    x = x[:, 0, :, :] * coef + x[:, 1, :, :] * coef + x[:, 2, :, :] * coef
    return x.repeat(1, 3, 1, 1)


def gray2color(x):
    x = x[:, 0, :, :]
    coef = 1 / 3
    base = coef ** 2 + coef ** 2 + coef ** 2
    return torch.stack((x * coef / base, x * coef / base, x * coef / base), 1)


def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = th.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n, c, h, 1, w, 1)
    out = out.view(n, c, scale * h, scale * w)
    return out



def fft2(X, size1 = 0, size2 = 0):


    if size1 == 0:
        X_fft1 = torch.fft.fft(X, dim=-1)
        X_fft2 = torch.fft.fft(X_fft1, dim=-2)

    else:
        X_fft1 = torch.fft.fft(X, dim=-1, n=size2)
        X_fft2 = torch.fft.fft(X_fft1, dim=-2, n=size1)

    return X_fft2

def ifft2(X_fft2, size1 = 0, size2 = 0):
    if size1 == 0:
        X_fft1 = torch.fft.ifft(X_fft2, dim=-2)
        X_fft_ifft = torch.fft.ifft(X_fft1, dim=-1)
    else:
        sf = 4
        k_size = sf * 4 + 3
        X_fft1 = torch.fft.ifft(X_fft2, dim=-2)
        X = torch.fft.ifft(X_fft1, dim=-1)
        X_fft_ifft = X[:, :, 0:size1, 0:size2]
    return abs(X_fft_ifft)


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)

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

def data_transform(config, X):
    if config['data']['uniform_dequantization']:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config['data']['gaussian_dequantization']:
        X = X + torch.randn_like(X) * 0.01

    if config['data']['rescaled']:
        X = 2 * X - 1.0
    elif config['data']['logit_transform']:
        X = logit_transform(X)

    # if hasattr(config, "image_mean"):
    #     return X - config.image_mean.to(X.device)[None, ...]

    return X



def save_kernel(k, savepath):
    k = move2cpu(k.squeeze())
    plt.clf()
    plt.axis('off')
    plt.imshow(k, vmin=0, vmax=k.max())
    # plt.colorbar(im, ax=ax[0, 0])
    plt.savefig(savepath)


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(img1, img2, is_kernel=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse)) if is_kernel else 20 * math.log10(255.0 / math.sqrt(mse))


def evaluation_image(hr, sr, sf):

    hr = tensor2im01(hr)
    sr = tensor2im01(sr)
    hr = rgb2ycbcr(hr / 255., only_y=True)
    sr = rgb2ycbcr(sr / 255., only_y=True)
    crop_border = sf**2
    cropped_sr = sr[crop_border:-crop_border, crop_border:-crop_border]
    hr_11 = (hr.shape[0]-cropped_sr.shape[0])//2
    hr_12 = hr.shape[0] - cropped_sr.shape[0] - hr_11
    hr_21 = (hr.shape[1] - cropped_sr.shape[1]) // 2
    hr_22 = hr.shape[1] - cropped_sr.shape[1] - hr_21
    cropped_hr = hr[hr_11:-hr_12, hr_21:-hr_22]
    im_psnr = calculate_psnr(cropped_hr * 255, cropped_sr * 255)
    im_ssim = calculate_ssim(cropped_hr * 255, cropped_sr * 255)

    return im_psnr, im_ssim


def fcn(num_input_channels=2, num_output_channels=3, num_hidden=1000):
    ''' fully-connected network as a kernel prior'''

    model = nn.Sequential()
    model.add(nn.Linear(num_input_channels, num_hidden, bias=True))
    model.add(nn.ReLU6())
    model.add(nn.Linear(num_hidden, num_output_channels))
    model.add(nn.Softmax())

    return model

def calculate_parameters(net):
    out = 0
    for param in net.parameters():
        out += param.numel()
    return out

def KL_Loss(kernel1, kernel2):
    KL_loss = 0.0
    eps =1e-7
    a = kernel1.size()
    [B, C, H, W] = kernel1.size()
    if B == 1 and C == 1:
        for i in range(H):
            for j in range(W):
                        KL_loss += (kernel1[:,:,i,j] + eps) * torch.log((kernel1[:,:,i,j] + eps) / (kernel2[:,:,i,j] + eps))
    return KL_loss

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    torch.manual_seed(1)
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input



def Compute_lossk(x, k, y):
    lossk = 1


    return lossk


def Move_X_forward(Y_fftm_ifftm, move_x = 9):

    size1 = Y_fftm_ifftm.shape[-2]
    size2 = Y_fftm_ifftm.shape[-1]

    Y_fftm_ifftm1 = torch.zeros_like(Y_fftm_ifftm)
    Y_fftm_ifftm2 = torch.zeros_like(Y_fftm_ifftm)
    Y_fftm_ifftm1[:, :, 0:(size1 - move_x), :] = Y_fftm_ifftm[:, :, move_x:size1, :]
    Y_fftm_ifftm1[:, :, (size1 - move_x):size1, :] = Y_fftm_ifftm[:, :, 0:move_x, :]
    Y_fftm_ifftm2[:, :, :, 0:(size2 - move_x)] = Y_fftm_ifftm1[:, :, :, move_x:size2]
    Y_fftm_ifftm2[:, :, :, (size2 - move_x):size2] = Y_fftm_ifftm1[:, :, :, 0:move_x]

    return Y_fftm_ifftm2


def Move_X_backward(Y_fftm_ifftm, move_x = 9):

    size1 = Y_fftm_ifftm.shape[-2]
    size2 = Y_fftm_ifftm.shape[-1]

    Y_fftm_ifftm1 = torch.zeros_like(Y_fftm_ifftm)
    Y_fftm_ifftm2 = torch.zeros_like(Y_fftm_ifftm)
    Y_fftm_ifftm1[:, :, move_x:size1, :] = Y_fftm_ifftm[:, :, 0:(size1 - move_x), :]
    Y_fftm_ifftm1[:, :, 0:move_x, :] = Y_fftm_ifftm[:, :, (size1 - move_x):size1, :]


    Y_fftm_ifftm2[:, :, :, move_x:size2] = Y_fftm_ifftm1[:, :, :, 0:(size2 - move_x)]
    Y_fftm_ifftm2[:, :, :, 0:move_x] = Y_fftm_ifftm1[:, :, :, (size2 - move_x):size2]

    return Y_fftm_ifftm2

def shift_cut(finalresult, xt, shift_h, shift_w, bias):
    if (shift_h == 0) and (shift_w != 0):  # 第一排的非第一张图片：需要在width方向割掉bias的边缘量
        finalresult[:, :, int(128 * shift_h):int(128 * shift_h) + 256,
        int(128 * shift_w + bias):int(128 * shift_w) + 256] = xt[:, :, :, bias:]
    elif (shift_h != 0) and (shift_w == 0):  # 第一列的非第一张图片：需要在height方向割掉bias的边缘量
        finalresult[:, :, int(128 * shift_h + bias):int(128 * shift_h) + 256,
        int(128 * shift_w):int(128 * shift_w) + 256] = xt[:, :, bias:, :]
    elif (shift_h == 0 and shift_w == 0):  # 第一张图像，不需要做任何处理
        finalresult[:, :, int(128 * shift_h):int(128 * shift_h) + 256, int(128 * shift_w):int(128 * shift_w) + 256] = xt
    else:  # 其余图像需要在height和width方向割掉bias的边缘量
        finalresult[:, :, int(128 * shift_h + bias):int(128 * shift_h) + 256,
        int(128 * shift_w + bias):int(128 * shift_w) + 256] = xt[:, :, bias:, bias:]
    return finalresult
