import os.path
import numpy as np
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        A_paths = make_dataset(self.dir_A)
        B_paths = make_dataset(self.dir_B)
        if self.opt.isTrain:
            self.length = min([len(A_paths), len(B_paths), self.opt.max_dataset_size])
            allow_repeats_A, allow_repeats_B = False, False
        else:
            self.length = min(max([len(A_paths), len(B_paths)]), self.opt.max_dataset_size)
            allow_repeats_A = self.length > len(A_paths)
            allow_repeats_B = self.length > len(B_paths)

        if opt.verbosity >= 1:
            print('Dataset for domain A contains %d images.' % len(A_paths))
            print('Dataset for domain B contains %d images.' % len(B_paths))
            print('Images limit %d.' % self.length)

        # make reproducible random choice of data subset
        state = np.random.get_state()
        np.random.seed(111)
        self.A_paths = sorted(np.random.choice(A_paths, self.length, replace=allow_repeats_A).tolist())
        self.B_paths = sorted(np.random.choice(B_paths, self.length, replace=allow_repeats_B).tolist())
        np.random.set_state(state)

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        def read_one_image(paths):
            im_path = paths[index]
            img = Image.open(im_path).convert('RGB')
            return im_path, self.transform(img)

        A_path, A_img = read_one_image(self.A_paths)
        B_path, B_img = read_one_image(self.B_paths)

        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return self.length

    def name(self):
        return 'UnalignedDataset'
