import os
from torch.utils.data.dataset import Dataset
from utils.degrade import *
import numpy as np
import h5py
from torch.utils.data import DataLoader
from einops import rearrange


class TrainSetLoader():
    def __init__(self, args):
        super(TrainSetLoader, self).__init__()
        self.args = args
        self.item_num = args.batch_size * 100

    def __getitem__(self, index):
        file_list = os.listdir(self.args.trainset_dir)
        file_idx = np.random.randint(len(file_list))
        file_name = [self.args.trainset_dir + file_list[file_idx]]
        with h5py.File(file_name[0], 'r') as hf:
            lf = np.array(hf.get('lf'))
            lf = np.transpose(lf, (4, 3, 0, 2, 1))
            lf = torch.from_numpy(lf)

        return lf

    def __len__(self):
        return self.item_num


def MultiTestSetDataLoader(args):
    data_list = os.listdir(args.testset_dir)
    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name)
        length_of_tests += len(test_Dataset)
        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=0, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name):
        super(TestSetDataLoader, self).__init__()
        self.args = args
        self.file_list = []
        self.testset_dir = args.testset_dir + data_name
        tmp_list = os.listdir(self.testset_dir)
        self.gen_LR = MultiDegrade(
            scale=self.args.upfactor,
            sig=args.sig,
            noise=args.noise,
        )
        self.file_list.extend(tmp_list)
        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = self.testset_dir + '/' + self.file_list[index]
        with h5py.File(file_name, 'r') as hf:
            lf = np.array(hf.get('lf'))
        lf = np.transpose(lf, (4, 3, 0, 2, 1))
        lf = torch.from_numpy(lf).to(self.args.device)
        U, V, H, W, _ = lf.shape
        lfimg = rearrange(lf, 'u v c h w -> (u v) c h w')
        lfimg_lr, sigma, noise_level = self.gen_LR(lfimg.unsqueeze(0))
        data = rearrange(lfimg_lr.squeeze(), '(u v) c h w -> u v c h w', u=U, v=V)
        a0 = (9 - self.args.angRes) // 2
        data = data[a0 : a0 + self.args.angRes, a0 : a0 + self.args.angRes, :, :, :]
        label = lf[a0: a0 + self.args.angRes, a0: a0 + self.args.angRes, :, :, :]
        return data, label, sigma, noise_level.squeeze(0)

    def __len__(self):
        return self.item_num
