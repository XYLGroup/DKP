import os
import argparse
import torch
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from model.util import read_image, im2tensor01, map2tensor, tensor2im01, analytic_kernel, kernel_shift, evaluation_dataset, modcrop
from config.configs import Config
from model.model import DIPDKP
import time
import datetime
from Settings import parameters_setting
import xlwt
import openpyxl
# for nonblind SR
sys.path.append('../')
from NonblindSR.usrnet import USRNet

'''
# ------------------------------------------------
# main.py for DIP-KP
# ------------------------------------------------
'''

def train(conf, lr_image, hr_image):
    ''' trainer for DIPDKP, etc.'''
    model = DIPDKP(conf, lr_image, hr_image)
    kernel, sr = model.train()
    return kernel, sr
def create_params(filename, args):
    ''' pass parameters to Config '''
    params = ['--model', args.model,
              '--input_image_path', args.input_dir + '/' + filename,
              # '--input_image_path', os.path.join(args.input_dir, filename),
              # '--output_dir_path', os.path.abspath(args.output_dir),
              # '--path_KP', os.path.abspath(args.path_KP),
              '--sf', args.sf]
    if args.SR:
        params.append('--SR')
    if args.real:
        params.append('--real')
    return params



def main():

    for Num_of_inits in range(1):
        I_loop_xs = []  # The number of the SISR outer iterations Q, default = 1, non-Gaussian kernel default = 1
        I_loop_xs.append(5)

        for Q in range(len(I_loop_xs)):
            I_loop_x = I_loop_xs[Q]
            I_loop_ks = []
            I_loop_ks.append(3)
            for P in range(len(I_loop_ks)):
                I_loop_k = I_loop_ks[P]
            # I_loop_k = 3  # The number of the meta-learning P, default = 3

                D_loops = []   # The number of the MCMC sampling times L, default >= 5
                D_loops.append(5)

                for j in range(len(D_loops)):
                    D_loop = D_loops[j]
                    datasets = []
                    datasets.append('Set5')

                    for k in range(len(datasets)):
                        dataset = datasets[k]
                        methods = []
                        methods.append('DIPDKP')

                        for i in range(len(methods)):
                            method = methods[i]

                            prog = argparse.ArgumentParser()    # 创建ArgumentParser对象
                            # 调用add_argument()方法添加参数
                            prog.add_argument('--model', type=str, default=method, help='models: DIPDKP.')
                            prog.add_argument('--dataset', '-d', type=str, default=dataset,
                                              help='dataset, e.g., Set5.')
                            prog.add_argument('--sf', type=str, default='4', help='The wanted SR scale factor')
                            prog.add_argument('--path-nonblind', type=str, default='../data/pretrained_models/usrnet_tiny.pth',
                                              help='path for trained nonblind model')
                            prog.add_argument('--SR', action='store_true', default=False, help='when activated - nonblind SR is performed')
                            prog.add_argument('--real', action='store_true', default=False, help='if the input is real image')

                            # 解析参数
                            args = prog.parse_args()

                            # load nonblind model
                            if args.SR:
                                netG = USRNet(n_iter=6, h_nc=32, in_nc=4, out_nc=3, nc=[16, 32, 64, 64],
                                              nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
                                netG.load_state_dict(torch.load(args.path_nonblind), strict=True)
                                netG.eval()
                                for key, v in netG.named_parameters():
                                    v.requires_grad = False
                                netG = netG.cuda()

                            args.input_dir = '../data/datasets/{}/DIPDKP_lr_x{}'.format(args.dataset, args.sf)
                            filesource = os.listdir(os.path.abspath(args.input_dir)) #
                            filesource.sort()
                            now_time = str(datetime.datetime.now())[:-10].replace(':', '-')

                            for filename in filesource[:]:
                                print(filename)
                                #setting the parameters
                                conf = Config().parse(create_params(filename, args))
                                conf, args = parameters_setting(conf, args, I_loop_x, I_loop_k, D_loop, method, filename, now_time)
                                lr_image = im2tensor01(read_image(os.path.join(args.input_dir, filename))).unsqueeze(0)


                                if args.real == False:
                                    hr_img = read_image(os.path.join(args.hr_dir, filename))
                                    # hr_img = modcrop(hr_img, conf.sf)
                                    hr_image = im2tensor01(hr_img).unsqueeze(0)
                                else:
                                    hr_image = torch.ones(lr_image.shape[0], lr_image.shape[1], lr_image.shape[2]*int(args.sf), lr_image.shape[3]*int(args.sf))

                                # crop the image to 960x960 due to memory limit
                                if 'DIV2K' in args.input_dir:
                                    crop_size = 800
                                    size_min = min(hr_image.shape[2], hr_image.shape[3])
                                    if size_min > crop_size:
                                        crop = int(crop_size / 2 / conf.sf)
                                        lr_image = lr_image[:, :, lr_image.shape[2] // 2 - crop: lr_image.shape[2] // 2 + crop,
                                                   lr_image.shape[3] // 2 - crop: lr_image.shape[3] // 2 + crop]
                                        hr_image = hr_image[:, :, hr_image.shape[2] // 2 - crop * 2: hr_image.shape[2] // 2 + crop * 2,
                                                   hr_image.shape[3] // 2 - crop * 2: hr_image.shape[3] // 2 + crop * 2]
                                    conf.IF_DIV2K = True
                                    conf.crop = crop

                                strat_time = time.time()
                                kernel, sr_dip = train(conf, lr_image, hr_image)
                                Runtime = time.time() - strat_time
                                # print("method:", method, "Runtime:", "%.2f" % Runtime)
                                plt.imsave(os.path.join(conf.output_dir_path, '%s.png' % conf.img_name), tensor2im01(sr_dip), vmin=0,
                                           vmax=1., dpi=1)

                                # nonblind SR
                                if args.SR:
                                    kernel = map2tensor(kernel)

                                    sr = netG(lr_image, torch.flip(kernel, [2, 3]), int(args.sf),
                                              (10 if args.real else 0) / 255 * torch.ones([1, 1, 1, 1]).cuda())
                                    plt.imsave(os.path.join(conf.output_dir_path, '%s.png' % conf.img_name), tensor2im01(sr), vmin=0,
                                               vmax=1., dpi=1)

                            if args.real == False:
                                image_psnr, im_ssim, kernel_psnr = evaluation_dataset(args.input_dir, conf.output_dir_path, conf)
                            # evaluation_dataset(args.input_dir, conf.output_dir_path, conf)
                            # prog.exit(0)

if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    main()
