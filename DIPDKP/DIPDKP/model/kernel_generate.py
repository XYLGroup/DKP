import numpy as np
from .util import kernel_move
import cv2
import torch
import sys

# noise
def make_gradient_filter():
    filters = np.zeros([4, 3, 3], dtype=np.float32)
    filters[0,] = np.array([[0, -1, 0],
                            [0, 1, 0],
                            [0, 0, 0]])

    filters[1,] = np.array([[-1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])

    filters[2,] = np.array([[0, 0, 0],
                            [-1, 1, 0],
                            [0, 0, 0]])

    filters[3,] = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [-1, 0, 0]])

    return torch.from_numpy(filters).cuda()



# kernel
def gen_kernel_random(k_size, scale_factor, min_var, max_var, noise_level, move_x, move_y):
    lambda_1 = min_var + np.random.rand() * (max_var - min_var);
    lambda_2 = min_var + np.random.rand() * (max_var - min_var);
    theta = np.random.rand() * np.pi
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    kernel = gen_kernel_fixed(k_size, scale_factor, lambda_1, lambda_2, theta, noise, move_x, move_y)

    return kernel


def gen_kernel_fixed(k_size, scale_factor, lambda_1, lambda_2, theta, noise, move_x, move_y):
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



def rotate(image, angle, center=None, scale=1.0):
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



def gen_kernel_motion_fixed(k_size, sf, lens, theta, noise):

    # kernel_size = min(sf * 4 + 3, 21)
    kernel_size = k_size[0]
    M = int((sf * 3 + 3) / 2)
    kernel_init = np.zeros([min(sf * 4 + 3, 21), min(sf * 4 + 3, 21)])
    # kernel_init[M-1:M+1,M-len:M-len] = 1
    kernel_init[M:M + 1, M - lens:M + lens + 1] = 1
    kernel = kernel_init + noise
    center = ((sf * 3 + 3) / 2, (sf * 3 + 3) / 2)
    kernel = rotate(kernel, theta, center, scale=1.0)

    kernel = kernel / np.sum(kernel)

    return kernel



def gen_kernel_random_motion(k_size, scale_factor, lens, noise_level):
    # lambda_1 = min_var + np.random.rand() * (max_var - min_var);
    # lambda_2 = min_var + np.random.rand() * (max_var - min_var);
    theta = np.random.rand() * 360  # np.pi
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    kernel = gen_kernel_motion_fixed(k_size, scale_factor, lens, theta, noise)

    return kernel


def ekp_kernel_generator(U, kernel_size, sf=4, shift='left'):
    '''
    Generate Gaussian kernel according to cholesky decomposion.
    \Sigma = M * M^T, M is a lower triangular matrix.
    Input:
        U: 2 x 2 torch tensor
        sf: scale factor
    Output:
        kernel: 2 x 2 torch tensor
    '''
    #  Mask
    mask = torch.tensor([[1.0, 0.0],
                         [1.0, 1.0]], dtype=torch.float32).to(U.device)
    M = U * mask

    # Set COV matrix using Lambdas and Theta
    INV_SIGMA = torch.mm(M.t(), M)

    # Set expectation position (shifting kernel for aligned image)
    if shift.lower() == 'left':
        MU = kernel_size // 2 - 0.5 * (sf - 1)
    elif shift.lower() == 'center':
        MU = kernel_size // 2
    elif shift.lower() == 'right':
        MU = kernel_size // 2 + 0.5 * (sf - 1)
    else:
        sys.exit('Please input corrected shift parameter: left , right or center!')

    # Create meshgrid for Gaussian
    X, Y = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size))
    Z = torch.stack((X, Y), dim=2).unsqueeze(3).type(torch.float32).to(U.device)  # k x k x 2 x 1

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - MU
    ZZ_t = ZZ.permute(0, 1, 3, 2)  # k x k x 1 x 2
    raw_kernel = torch.exp(-0.5 * torch.squeeze(ZZ_t.matmul(INV_SIGMA).matmul(ZZ)))

    # Normalize the kernel and return
    kernel = raw_kernel / torch.sum(raw_kernel)  # k x k
    return kernel.unsqueeze(0).unsqueeze(0)