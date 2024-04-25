
import numpy as np
import torch
import cv2

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

def gen_kernel_line_motion(kernel_size, sf, lens = 3, noise_level = 0):
    # lambda_1 = min_var + np.random.rand() * (max_var - min_var);
    # lambda_2 = min_var + np.random.rand() * (max_var - min_var);
    # theta = np.random.rand() * 360 # np.pi

    theta = np.random.randint(int(180/self.random_angle)) * self.random_angle

    noise = -noise_level + np.random.rand(self.kernel_size) * noise_level * 2

    kernel = self.gen_kernel_motion_line_fixed(self.kernel_size, self.sf, lens, theta, noise)

    kernel = torch.from_numpy(kernel).type(torch.FloatTensor).to(torch.device('cuda')).unsqueeze(0).unsqueeze(0)

    return kernel





def obtain_kernel(kernel_type):
    if kernel_type == 'Motion_line':
        return gen_kernel_line_motion()



def main():