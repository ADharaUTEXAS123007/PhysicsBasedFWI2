import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torch


class UnalignedBD2TestDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain C '/path/to/data/trainC' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testC' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of CaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot,  'testB')  # create a path '/path/to/data/trainA'
        self.dir_C = os.path.join(opt.dataroot,  'testD')  # create a path '/path/to/data/trainC'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.C_paths = sorted(make_dataset(self.dir_C, opt.max_dataset_size))    # load images from '/path/to/data/trainC'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.C_size = len(self.C_paths)  # get the size of dataset C
        ctoA = self.opt.direction == 'CtoA'
        input_nc = self.opt.output_nc if ctoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if ctoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_C = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, C, A_paths and C_paths
            A (tensor)       -- an image in the input domain
            C (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            C_paths (str)    -- image paths
        """
        #A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        #if self.opt.serial_batches:   # make sure index is within then range
        #    index_C = index % self.C_size
        #else:   # randomize the index for domain C to avoid fixed pairs.
        #    index_C = random.randint(0, self.C_size - 1)
        A_path = self.A_paths[index]
        C_path = self.C_paths[index]
        A_img = np.load(A_path)
        C_img = np.load(C_path)


        #C_img = np.expand_dims(C_img,0)
        A_img = np.expand_dims(A_img,0)
        C_img = np.expand_dims(C_img,0)
        A = torch.from_numpy(A_img)
        A = A.float()
        C = torch.from_numpy(C_img)
        C = C.float()

        #print("AC size")
        #print(A.size())
        #print(C.size())

        #print("after rot")
        #r = random.randint(0,3)
        #A = torch.rot90(A, r, [1, 2])
        #C = torch.rot90(C, r, [1, 2])
        #print(A.size())
        #print(C.size())

        #print("after flip")
        #r = random.randint(0,1)
        #if (r == 0):
        #    A = torch.flip(A,[1,2])
        #    C = torch.flip(C,[1,2])
        #print(A.size())
        #print(C.size())
        # apply image transformation
        #A = self.transform_A(A_img)
        #C = self.transform_C(C_img)
        #print("sizes")
        #print(A.size())
        #print(C.size())

        return {'A': A, 'C': C, 'A_paths': A_path, 'C_paths': C_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.C_size)
