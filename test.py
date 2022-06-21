import argparse
from utils.utility import *
from model.DAnet import Net
import numpy as np
import imageio
import torch
import os
from einops import rearrange


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--upfactor", type=int, default=4, help="upscale factor")
    parser.add_argument('--crop', type=bool, default=True, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--patchsize", type=int, default=32, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--minibatch", type=int, default=20, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument('--input_dir', type=str, default='./input/')
    parser.add_argument('--save_path', type=str, default='./output/')

    return parser.parse_args()


def demo_test(cfg, sigma_range, noise_range):

    net = Net(cfg.upfactor, cfg.angRes)
    net.to(cfg.device)
    model = torch.load('./log/LF-DAnet_4xSR_paper.tar', map_location={'cuda:1': cfg.device})
    net.load_state_dict(model['state_dict'])

    scene_list = os.listdir(cfg.input_dir)

    for scenes in scene_list:
        for sigma in sigma_range:
            for noise_level in noise_range:
                print('Scene: ' + scenes + '\t' + 'blur=' + str(sigma * 4) + '\t' + 'noise=' + str(noise_level * 75))
                temp = imageio.imread(cfg.input_dir + scenes + '/view_01_01.png')
                lf_rgb_lr = np.zeros(shape=(cfg.angRes, cfg.angRes, temp.shape[0], temp.shape[1], 3))

                for u in range(cfg.angRes):
                    for v in range(cfg.angRes):
                        temp = imageio.imread(cfg.input_dir + scenes + '/view_%.2d_%.2d.png' % (u+1, v+1))
                        lf_rgb_lr[u, v, :, :, :] = temp

                data = torch.from_numpy(lf_rgb_lr.astype('float32')) / 255.0
                data = rearrange(data, 'u v h w c -> u v c h w')

                if cfg.crop == False:
                    lf_rgb_sr = net(data.unsqueeze(0).to(cfg.device))

                else:
                    patchsize = cfg.patchsize
                    stride = patchsize // 2
                    sub_lfs = LFdivide(data, patchsize, stride)
                    n1, n2, u, v, c, h, w = sub_lfs.shape
                    sub_lfs = rearrange(sub_lfs, 'n1 n2 u v c h w -> (n1 n2) u v c h w')
                    mini_batch = cfg.minibatch
                    num_inference = (n1 * n2) // mini_batch
                    with torch.no_grad():
                        out_lfs = []
                        for idx_inference in range(num_inference):
                            torch.cuda.empty_cache()
                            input_lfs = sub_lfs[idx_inference * mini_batch: (idx_inference + 1) * mini_batch, :, :, :]
                            blur = torch.ones(input_lfs.shape[0], u, v).to(cfg.device) * (sigma + 1e-4)
                            noise = torch.ones(input_lfs.shape[0], u, v).to(cfg.device) * (noise_level + 1e-4)
                            out_temp = net((input_lfs.to(cfg.device), blur.unsqueeze(1), noise.unsqueeze(1)))
                            out_lfs.append(out_temp.to('cpu'))

                        if (n1 * n2) % mini_batch:
                            input_lfs = sub_lfs[(idx_inference + 1) * mini_batch:, :, :, :]
                            blur = torch.ones(input_lfs.shape[0], u, v).to(cfg.device) * (sigma + 1e-4)
                            noise = torch.ones(input_lfs.shape[0], u, v).to(cfg.device) * (noise_level + 1e-4)
                            out_temp = net((input_lfs.to(cfg.device), blur.unsqueeze(1), noise.unsqueeze(1)))
                            out_lfs.append(out_temp.to('cpu'))

                    out_lfs = torch.cat(out_lfs, dim=0)
                    out_lfs = rearrange(out_lfs, '(n1 n2) u v c h w -> n1 n2 u v c h w', n1=n1, n2=n2)
                    outLF = LFintegrate(out_lfs, patchsize * cfg.upfactor, patchsize * cfg.upfactor // 2)
                    lf_rgb_sr = outLF[:, :, :, 0: data.shape[3] * cfg.upfactor, 0: data.shape[4] * cfg.upfactor]
                    lf_rgb_sr = rearrange(lf_rgb_sr, 'u v c h w -> u v h w c')

                lf_rgb_sr = 255 * lf_rgb_sr.data.cpu().numpy()
                lf_rgb_sr = np.clip(lf_rgb_sr, 0, 255)

                output_path = cfg.save_path + scenes
                if not (os.path.exists(output_path)):
                    os.makedirs(output_path)

                #imageio.imwrite(output_path + '/LF-DAnet_blur_' + str(sigma * 4) + '_noise_' + str(noise_level * 75) + '.png', np.uint8(lf_rgb_sr[2, 2, :, :, :]))

                for u in range(cfg.angRes):
                    for v in range(cfg.angRes):
                        imageio.imwrite(output_path + '/LF-DAnet_view_%.2d_%.2d.png' % (u+1, v+1), np.uint8(lf_rgb_sr[u, v, :, :, :]))

    print('All Finished! \n')


if __name__ == '__main__':
    cfg = parse_args()
    # sigma_range = [0/4, 1/4, 2/4, 3/4, 4/4]
    # noise_range = [0/75, 15/75, 30/75, 45/75, 60/75]
    sigma_range = [2/4]
    noise_range = [30/75]
    demo_test(cfg, sigma_range, noise_range)
