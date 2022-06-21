import argparse
import torch.backends.cudnn as cudnn
from utils.utility import *
from utils.dataloader import *
from model.DAnet import Net


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--model_name', type=str, default='LF-DAnet')
parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
parser.add_argument("--upfactor", type=int, default=4, help="upscale factor")
parser.add_argument('--model_path', type=str, default='./log/LF-DAnet_4xSR_paper.tar')
parser.add_argument('--crop', type=bool, default=True)
parser.add_argument("--patchsize_test", type=int, default=32, help="patchsize of LR images for inference")
parser.add_argument("--minibatch_test", type=int, default=10, help="size of minibatch for inference")
parser.add_argument('--testset_dir', type=str, default='../Data/Validation_MDSR/')

args = parser.parse_args()


def train(args):
    net = Net(factor=args.upfactor, angRes=args.angRes)
    net.to(args.device)
    cudnn.benchmark = True

    model = torch.load(args.model_path, map_location={'cuda:0': args.device})
    net.load_state_dict(model['state_dict'], strict=False)

    for noise in [0, 15, 50]:
        args.noise = noise
        for sig in [0, 1.5, 3, 4.5]:
            args.sig = sig
            test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
            for index, test_name in enumerate(test_Names):
                torch.cuda.empty_cache()
                test_loader = test_Loaders[index]
                psnr_epoch_test, ssim_epoch_test = valid(test_loader, net)
                print('Dataset--%15s,\t noise--%f, \t sig---%f, \t PSNR--%f, \t SSIM---%f' % (
                    test_name, args.noise, sig, psnr_epoch_test, ssim_epoch_test))


def valid(test_loader, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label, sigma, noise_level) in (enumerate(test_loader)):


        gt_blur = sigma.unsqueeze(1).unsqueeze(1).repeat(1, 1, args.angRes, args.angRes) / 4
        gt_noise = noise_level.repeat(1, 1, args.angRes, args.angRes)

        if args.crop == False:
            with torch.no_grad():
                outLF = net(data)
                outLF = outLF.squeeze()
        else:
            patch_size = args.patchsize_test
            data = data.squeeze()
            sub_lfs = LFdivide(data, patch_size, patch_size // 2)

            n1, n2, u, v, c, h, w = sub_lfs.shape
            sub_lfs =  rearrange(sub_lfs, 'n1 n2 u v c h w -> (n1 n2) u v c h w')
            mini_batch = args.minibatch_test
            num_inference = (n1 * n2) // mini_batch
            with torch.no_grad():
                out_lfs = []
                for idx_inference in range(num_inference):
                    torch.cuda.empty_cache()
                    input_lfs = sub_lfs[idx_inference * mini_batch : (idx_inference+1) * mini_batch, :, :, :, :, :]
                    lf_out = net((input_lfs.to(args.device), gt_blur.repeat(input_lfs.shape[0], 1, 1, 1), gt_noise.repeat(input_lfs.shape[0], 1, 1, 1)))
                    out_lfs.append(lf_out)
                if (n1 * n2) % mini_batch:
                    torch.cuda.empty_cache()
                    input_lfs = sub_lfs[(idx_inference+1) * mini_batch :, :, :, :, :, :]
                    lf_out = net((input_lfs.to(args.device), gt_blur.repeat(input_lfs.shape[0], 1, 1, 1), gt_noise.repeat(input_lfs.shape[0], 1, 1, 1)))
                    out_lfs.append(lf_out)

            out_lfs = torch.cat(out_lfs, dim=0)
            out_lfs = rearrange(out_lfs, '(n1 n2) u v c h w -> n1 n2 u v c h w', n1=n1, n2=n2)
            outLF = LFintegrate(out_lfs, patch_size * args.upfactor, patch_size * args.upfactor // 2)
            outLF = outLF[:, :, :, 0 : data.shape[3] * args.upfactor, 0 : data.shape[4] * args.upfactor]

        psnr, ssim = cal_metrics(label.squeeze(), outLF)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test


def augmentation(x, y):
    if random.random() < 0.5:  # flip along U-H direction
        x = torch.flip(x, dims=[1, 4])
        x = torch.flip(x, dims=[1, 4])
    if random.random() < 0.5:  # flip along W-V direction
        y = torch.flip(y, dims=[2, 5])
        y = torch.flip(y, dims=[2, 5])
    if random.random() < 0.5: # transpose between U-V and H-W
        x = x.permute(0, 2, 1, 3, 5, 4)
        y = y.permute(0, 2, 1, 3, 5, 4)

    "random color shuffling"
    if random.random() < 0.5:
        color = [0, 1, 2]
        random.shuffle(color)
        x, y = x[:, :, :, color, :, :], y[:, :, :, color, :, :]

    return x, y


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    train(args)
