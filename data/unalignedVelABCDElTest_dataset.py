import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torch


class UnalignedVelABCDElTestDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, 'testA')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, 'testB')  # create a path '/path/to/data/trainB'
        self.dir_C = os.path.join(opt.dataroot, 'testC')  # create a path '/path/to/data/trainB'
        self.dir_D = os.path.join(opt.dataroot, 'testD')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.C_paths = sorted(make_dataset(self.dir_C, opt.max_dataset_size)) 
        self.D_paths = sorted(make_dataset(self.dir_D, opt.max_dataset_size)) 
        
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.C_size = len(self.C_paths)  # get the size of dataset C
        self.D_size = len(self.D_paths)
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        #A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        #if self.opt.serial_batches:   # make sure index is within then range
        #    index_B = index % self.B_size
        #else:   # randomize the index for domain B to avoid fixed pairs.
        #    index_B = random.randint(0, self.B_size - 1)
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        C_path = self.C_paths[index]
        D_path = self.D_paths[index]
        
        A_img = np.load(A_path)
        B_img = np.load(B_path)
        C_img = np.load(C_path)
        D_img = np.load(D_path)
        #B_img = (B_img - 2000)/(4500 - 2000)
        #B_img = (B_img - 1600)/(2300 - 1600)
        #C_img = (C_img - 1600)/(2300 - 1600)
        A_img = A_img
        B_img = B_img
        A_img = A_img
        B_img = B_img/10.0
        #B_img[2,:,:] = B_img[2,:,:]*10
        
        C_img = C_img/10.0
        #C_img[2,:,:] = C_img[2,:,:]*10
        D_img = D_img
        #r = random.randint(0,1)
        #if (r==0):
        #    A_img = -1*A_img
        #    B_img = -1*B_img


        #A_img = np.expand_dims(A_img,0)
        #B_img = np.expand_dims(B_img,0)
        #C_img = np.expand_dims(C_img,0)
        #D_img = np.expand_dims(D_img,0)
        A = torch.from_numpy(A_img)
        #A = torch.abs(A)
        A = A.float()
        B = torch.from_numpy(B_img)
        #B = torch.abs(B)
        B = B.float()
        
        C = torch.from_numpy(C_img)
        C = C.float()
        
        D = torch.from_numpy(D_img)
        D = D.float()

        #print("AB size")
        #print(A.size())
        #print(B.size())

        #print("after rot")
        #r = random.randint(0,3)
        #A = torch.rot90(A, r, [1, 2])
        #B = torch.rot90(B, r, [1, 2])
        #print(A.size())
        #print(B.size())

        #print("after flip")
        #r = random.randint(0,1)
        #if (r == 0):
        #    A = torch.flip(A,[1,2])
        #    B = torch.flip(B,[1,2])
        #print(A.size())
        #print(B.size())
        # apply image transformation
        #A = self.transform_A(A_img)
        #B = self.transform_B(B_img)
        #print("sizes")
        #print(A.size())
        #print(B.size())

        return {'A':A, 'B': B, 'C':C, 'D':D, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
