import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    """Dataset class for paired training (x_ROI, y_T).

    It assumes that the directory structure is:
        dataroot/trainA  (input RGB ROI images)
        dataroot/trainB  (pseudo-thermal targets)
        dataroot/testA   (validation RGB ROI)
        dataroot/testB   (validation targets)
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  # e.g. trainA
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  # e.g. trainB

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))

        assert self.opt.load_size >= self.opt.crop_size
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        A_img = Image.open(A_path).convert("RGB")   # input ROI
        B_img = Image.open(B_path).convert("RGB")   # target pseudo-thermal

        transform_params = get_params(self.opt, A_img.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A_img)
        B = B_transform(B_img)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        return min(len(self.A_paths), len(self.B_paths))
