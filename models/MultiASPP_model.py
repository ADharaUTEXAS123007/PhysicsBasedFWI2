import torch
from .base_model import BaseModel
from . import networks
import numpy as np
from skimage import metrics


class MultiASPPModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
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

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='MultiASPP', dataset_mode='unaligned2', ngf='32')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=1.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1', 'V_L1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['fake_B1T', 'real_B1T','fake_B2T','real_B2T']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.criterionRMSE = torch.nn.MSELoss()
        else:
            print("----test data----")
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionRMSE = torch.nn.MSELoss()

            


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        [self.fake_B1,self.fake_B2] = self.netG(self.real_A)  # G(A)
        #self.fake_B1 = torch.unsqueeze(self.fake_B[:,0,:,:],1)
        #self.fake_B2 = torch.unsqueeze(self.fake_B[:,1,:,:],1)
        self.real_B1 = torch.unsqueeze(self.real_B[:,0,:,:],1)
        self.real_B2 = torch.unsqueeze(self.real_B[:,1,:,:],1)

    def forwardT(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        [self.fake_B1T,self.fake_B2T] = self.netG(self.real_A)  # G(A)
        self.real_BT = self.real_B
        #self.fake_B1T = torch.unsqueeze(self.fake_BT[:,0,:,:],1)
        #self.fake_B2T = torch.unsqueeze(self.fake_BT[:,1,:,:],1)
        self.real_B1T = torch.unsqueeze(self.real_BT[:,0,:,:],1)
        #print(np.shape(self.real_BT))
        self.real_B2T = torch.unsqueeze(self.real_BT[:,1,:,:],1)
        #self.real_DT = torch.unsqueeze(self.real_A[:,2,:,:],1)
        #print("real_DT")
        #print(np.shape(self.real_A))

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        # Second, G(A) = B
        self.loss_G_L1 = (self.criterionL1(self.fake_B1, self.real_B1))*10 + (self.criterionL1(self.fake_B2, self.real_B2))*10
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def compute_loss_only(self):
        lossL1 = self.criterionL1(self.fake_B1T,self.real_B1T)*10 + self.criterionL1(self.fake_B2T,self.real_B2T)*10
        self.loss_V_L1 = lossL1
        print("Loss L1 : "+ str(lossL1.cpu().float().numpy()))
        #file1 = open("/glb/data/eptr_am_2/Arnab/volumes/training_data/events/HorizonWindow/att_volumes/slices/finalTest/loss.txt","a")
        #file1.write(str(lossL1.cpu().float().numpy()))
        #file1.close()
        #lossRMSE = torch.sqrt(self.criterionRMSE(self.fake_BT,self.real_BT)) 
        #self.loss_V_L1 = lossRMSE
        #print("Loss RMSE : "+str(lossRMSE.cpu().float().numpy()))
        #lossSSIM = metrics.structural_similarity(np.squeeze(self.fake_B.cpu().float().numpy()),np.squeeze(self.real_B.cpu().float().numpy()) )
        #print("Loss SSIM :"+str(lossSSIM))
        #loss_RMSE = torch.nn.MSELoss(self.fake_B,self.real_B)
        #loss_RMSE = np.sqrt(loss_RMSE.cpu().float().numpy())
        #print("RMSE loss :" + loss_RMSE)
        pass

