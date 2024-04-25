

import os
import argparse
import torch
import torch as th
import torch.nn.functional as F
import time
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util
import torchvision.transforms as transforms
from PIL import Image
from guided_diffusion.utils2 import read_image, im2tensor01, get_dataset, evaluation_image
import tqdm
import torch.utils.data as data
import yaml
import xlwt
from thop import profile
from guided_diffusion.scheduler import get_schedule_jump
from guided_diffusion.unet import UNetModel
from torchstat import stat

# Workaround
try:
    import ctypes

    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)  # noqa: E402


def main(conf: conf_mgt.Default_Conf, args, val_loader, dataset, dataset_name):
    print("Start", conf['name'])

    device = dist_util.dev(conf.get('device'))

    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # def calculate_parameters(net):
    #     out = 0
    #     for param in net.parameters():
    #         out += param.numel()
    #     return out

    # log_str = 'Number of parameters in Diffusion: {:.2f}K'
    # print(log_str.format(calculate_parameters(model) / 1000))

    show_progress = conf.show_progress

    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(
                conf.classifier_path), map_location="cpu")
        )

        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    else:
        cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    avg_psnr = 0.0
    avg_ssim = 0.0
    pbar = tqdm.tqdm(val_loader)

    num = -1
    avg_psnr = 0
    avg_ssim = 0
    avg_k_psnr = 0

    if not os.path.exists(('results/' + args.get('save_path'))):
        # 检查该路径是否存在
        os.makedirs(('results/' + args.get('save_path')), exist_ok=True)

    xlwt_save_path = os.path.join('results/' + args.get('save_path') + '/test_results.xlsx')

    wb = xlwt.Workbook()  # 一个实例
    sheet = wb.add_sheet("Sheet1")  # 工作簿名称
    style = "font:colour_index black;"  # 设置样式
    black_style = xlwt.easyxf(style)
    sheet.write(0, 1, "image PSNR")
    sheet.write(0, 2, "image SSIM")
    sheet.write(0, 3, "kernel PSNR")

    for i in range(1, 100):
        sheet.write(i, 0, str(i))  # 不设置格式
        wb.save(xlwt_save_path)  # 最后一定要保存，否则无效

    col1 = 1
    col2 = 2
    col3 = 3

    for x_orig, classes in pbar:

        # all_images = []
        num = num + 1

        dset = 'eval'

        Y_path = val_loader.dataset.dataset.imgs[num][0]
        K_path = Y_path.replace('HR_256\\HR_256', 'DIPFKP_gt_k_x4')
        K_mat_path = K_path.replace('.png', '.mat')

        eval_name = conf.get_default_eval_name()

        dl = conf.get_dataloader(dset=dset, dsName=eval_name)

        for batch in iter(dl):

            for k in batch.keys():
                if isinstance(batch[k], th.Tensor):
                    batch[k] = batch[k].to(device)

            model_kwargs = {}

            X_path = dataset.imgs[num - 1][0]
            x_orig = im2tensor01(read_image(X_path)).unsqueeze(0)
            x_orig = x_orig.to(device)

            gt = x_orig

            # gt = Image.open(args.get("path_y")).convert('RGB')
            # data_transform = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # ])
            # # print("gt.size:",gt.size)
            # gt = data_transform(gt).unsqueeze(0).to("cuda")

            model_kwargs["gt"] = gt
            model_kwargs['scale'] = args.get('scale')
            model_kwargs['deg'] = args.get('deg')
            model_kwargs['resize_y'] = args.get('resize_y')
            model_kwargs['sigma_y'] = args.get('sigma_y')
            model_kwargs['save_path'] = args.get('save_path')

            model_kwargs['path_y'] = args.get('path_y')
            model_kwargs['dip_lr'] = args.get('dip_lr')
            model_kwargs['kp_lr'] = args.get('kp_lr')
            model_kwargs['kernel_type'] = args.get('kernel_type')
            model_kwargs['k_mat_path'] = K_mat_path

            model_kwargs['dataset_name'] = dataset_name
            model_kwargs['image_num'] = num

            gt_keep_mask = batch.get('gt_keep_mask')
            if gt_keep_mask is not None:
                model_kwargs['gt_keep_mask'] = gt_keep_mask

            batch_size = model_kwargs["gt"].shape[0]

            if conf.cond_y is not None:
                classes = th.ones(batch_size, dtype=th.long, device=device)
                model_kwargs["y"] = classes * conf.cond_y
            else:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(batch_size,), device=device
                )
                model_kwargs["y"] = classes

            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * args.get("class")

            sample_fn = (
                diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
            )

            # flops, params = profile(model=model, inputs=(torch.randn(1, 3, 256, 256),))
            # flops_G = flops/(1024*1024*1024)
            # params_M = params/(1024*1024)


            # stat(model, ( 3, 256, 256))

            result = sample_fn(
                model_fn,
                (batch_size, 3, conf.image_size, conf.image_size),
                clip_denoised=conf.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=device,
                progress=show_progress,
                return_all=True,
                conf=conf
            )

        image_psnr, image_ssim = evaluation_image(result["sample"], result["gt"], sf=4)

        print('final evalution: PSNR:{:.2f}, SSIM:{:.4f}, Kernel PSNR:{:.2f}'.format(image_psnr, image_ssim, result['kernel_psnr']))

        sheet.write(num+1, col1, '%.2f' % image_psnr, black_style)
        sheet.write(num+1, col2, '%.4f' % image_ssim, black_style)
        sheet.write(num+1, col3, '%.2f' % result['kernel_psnr'], black_style)
        wb.save(xlwt_save_path)

        avg_psnr += image_psnr
        avg_ssim += image_ssim
        avg_k_psnr += result['kernel_psnr']

        print("sampling complete")

    final_psnr = avg_psnr / (num + 1)
    final_ssim = avg_ssim / (num + 1)
    final_k_psnr = avg_k_psnr / (num + 1)

    print('Average result: ave_PSNR/ave_SSIM:{:.2f}/{:.4f}, Kernel PSNR:{:.2f}'.format(final_psnr, final_ssim, final_k_psnr))
    sheet.write(num + 2, col1, '%.2f' % final_psnr, black_style)
    sheet.write(num + 2, col2, '%.4f' % final_ssim, black_style)
    sheet.write(num + 2, col3, '%.2f' % final_k_psnr, black_style)
    wb.save(xlwt_save_path)


if __name__ == "__main__":
    dataset_name = 'Set5'  # Set5  Set14  Urban100  BSD100  CVPR-visual  CVPR-real CVPR-intermediate
    sf = 4
    kernel_type = 'Gaussian'  #  'Gaussian' 'Motion' , 'Motion_4'  'Motion_line'  read_test
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, default="confs/inet256.yml")  # inet256, face256
    parser.add_argument('--deg', type=str, required=False, default="deblur")  # deblur, inpainting, sr_averagepooling
    parser.add_argument('--sigma_y', type=float, required=False, default=0.)
    parser.add_argument('-i', "--save_path", type=str, required=False,
                        default="test/{}/{}_x{}/".format(dataset_name, kernel_type, sf))  # output SR fold
    parser.add_argument('--path_y', type=str, required=False,
                        default="deblur/{}/HR_256".format(dataset_name))  # input LR fold  # _256
    parser.add_argument('--scale', type=int, required=False, default=sf)
    parser.add_argument('--resize_y', default=False, action='store_true')
    parser.add_argument('--kp_lr', type=float, required=False, default=5e-3)  #  5e-4 for Gaussian   5e-3 for Motion line
    parser.add_argument('--kernel_type', type=str, default=kernel_type)  # 1e-3 for Gaussian   4e-3 for Motion line

    """
    SR scales should be divisible by 256, e.g., 2, 4, 8, 16 ...
    """


    """
    resize y to the same shape with the desired result
    """

    """
    orange.png
    bear.png
    flamingo.png
    kimono.png
    zebra.png
    """

    """
    950:orange
    294:brown bear
    130:flamingo
    614:kimono
    340:zebra
    """
    parser.add_argument('--class', type=int, required=False, default=950)

    parser.add_argument('--dip_lr', type=float, required=False, default=5e-3)


    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )

    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('config')))

    args['exp'] = os.getcwd()

    dataset, test_dataset = get_dataset(args, conf_arg)

    print(f'Dataset has size {len(test_dataset)}')

    g = torch.Generator()
    g.manual_seed(1234)

    val_loader = data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        worker_init_fn=None,
        generator=g,
    )

    subset_start = 0
    print(f'Start from {subset_start}')
    idx_init = subset_start
    idx_so_far = subset_start

    main(conf_arg, args, val_loader, dataset, dataset_name)
