import torch
from .base_model import BaseModel
from . import networks
import numpy as np
from skimage import metrics


class UnetSSIMACModel(BaseModel):
    """ This class implements the unet model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    Cy default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For unet, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_128', dataset_mode='unalignedAC2', ngf='16')
        return parser

    def __init__(self, opt):
        """Initialize the unet class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN','G_L1','V_L1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['fake_C_1', 'real_C_1', 'fake_C_2', 'real_C_2', 'fake_C_1T', 'real_C_1T', 'fake_C_2T', 'real_C_2T']
        #self.visual_names = []
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>

        self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.criterionRMSE = torch.nn.MSELoss()
            
        else:
            print("----test data----")
            self.criterionL1loss = torch.nn.L1Loss()
            self.criterionRMSE = torch.nn.MSELoss()

            


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain C.
        """
        AtoC = self.opt.direction == 'AtoC'
        self.real_A = input['A' if AtoC else 'C'].to(self.device)
        self.real_C = input['C' if AtoC else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoC else 'C_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_C = self.netG(self.real_A)  # G(A)
        self.fake_C_1 = self.fake_C[:,0,:,:]
        self.fake_C_1 = torch.unsqueeze(self.fake_C_1,1)
        self.fake_C_2 = self.fake_C[:,1,:,:]
        self.fake_C_2 = torch.unsqueeze(self.fake_C_2,1)
        self.real_C_1 = self.real_C[:,0,:,:]
        self.real_C_1 = torch.unsqueeze(self.real_C_1,1)
        self.real_C_2 = self.real_C[:,1,:,:]
        self.real_C_2 = torch.unsqueeze(self.real_C_2,1)

    def forwardT(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_C = self.netG(self.real_A)  # G(A)
        self.fake_C_1T = self.fake_C[:,0,:,:]
        self.fake_C_1T = torch.unsqueeze(self.fake_C_1T,1)
        self.fake_C_2T = self.fake_C[:,1,:,:]
        self.fake_C_2T = torch.unsqueeze(self.fake_C_2T,1)
        self.real_C_1T = self.real_C[:,0,:,:]
        self.real_C_1T = torch.unsqueeze(self.real_C_1T,1)
        self.real_C_2T = self.real_C[:,1,:,:]
        self.real_C_2T = torch.unsqueeze(self.real_C_2T,1)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G_L1 = self.criterionL1(self.fake_C, self.real_C)*100
        #print("Loss G_1 :"+str(self.loss_G_L1))
        # combine loss and calculate gradients
        self.loss_G_GAN = self.loss_G_L1
        self.loss_G = self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def compute_loss_only(self):
        lossL1 = self.criterionL1(self.fake_C,self.real_C)*100
        self.loss_V_L1 = lossL1
        print("Loss L1 : "+ str(lossL1.cpu().float().numpy()))
        lossRMSE = torch.sqrt(self.criterionRMSE(self.fake_C, self.real_C))*100
        #self.loss_V_L1 = lossRMSE
        print("Loss RMSE : "+str(lossRMSE.cpu().float().numpy()))
        #lossSSIM = metrics.structural_similarity(np.squeeze(self.fake_C.cpu().float().numpy()),np.squeeze(self.real_C.cpu().float().numpy()) )
        #print("Loss SSIM :"+str(lossSSIM))
        #loss_RMSE = torch.nn.MSELoss(self.fake_C,self.real_C)
        #loss_RMSE = np.sqrt(loss_RMSE.cpu().float().numpy())
        #print("RMSE loss :" + loss_RMSE)
        pass

