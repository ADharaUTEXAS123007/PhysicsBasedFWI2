import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import sys
# sys.path.append('/disk/student/adhara/Fall2021/deepwave/.')
# sys.path.append('/disk/student/adhara/Fall2021/deepwave/deepwave/.')
import deepwave
import multiprocessing
from joblib import Parallel, delayed
#import ray
import time
import os
#import scipy
import torchgeometry as tgm

#from skimage import metrics


class Auto2Model(BaseModel):
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
        parser.set_defaults(norm='batch', netG='Auto',
                            dataset_mode='unalignedVelABCD2', ngf='32')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float,
                                default=1.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        print("number of cuda devices:", torch.cuda.device_count())
        #for i in range(3):
        #torch.cuda.set_device(1)

        # torch.cuda.set_device(2)
        #del self.device
        # print(self.device)
        # Start Ray.
        #os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3,4,5,6,7"
        #ray.init(num_cpus=48,num_gpus=7)

        self.device1 = torch.device('cuda:{}'.format(
             self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        # self.device2 = torch.device('cuda:{}'.format(
        #     self.gpu_ids[1])) if self.gpu_ids else torch.device('cpu')
        # self.device3 = torch.device('cuda:{}'.format(
        #     self.gpu_ids[2])) if self.gpu_ids else torch.device('cpu')
        # self.device4 = torch.device('cuda:{}'.format(
        #     self.gpu_ids[3])) if self.gpu_ids else torch.device('cpu')
        # self.device5 = torch.device('cuda:{}'.format(
        #     self.gpu_ids[4])) if self.gpu_ids else torch.device('cpu')
        # self.device6 = torch.device('cuda:{}'.format(
        #     self.gpu_ids[5])) if self.gpu_ids else torch.device('cpu')
        # self.device7 = torch.device('cuda:{}'.format(
        #     self.gpu_ids[6])) if self.gpu_ids else torch.device('cpu')
        # self.device8 = torch.device('cuda:{}'.format(
        #     self.gpu_ids[7])) if self.gpu_ids else torch.device('cpu')
        
        
        # for i in range(2):
        #    variable = str(self.device)+str(i+1)
        #print("variable name :",variable)
        #    locals()[variable] = torch.device('cuda:{}'.format(self.gpu_ids[i])) if self.gpu_ids else torch.device('cpu')

        #self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        #self.device2 = torch.device('cuda:{}'.format(self.gpu_ids[1])) if self.gpu_ids else torch.device('cpu')
        #self.device3 = torch.device('cuda:{}'.format(self.gpu_ids[2])) if self.gpu_ids else torch.device('cpu')
        # self.device4 =
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_MSE', 'M_MSE', 'V_MSE']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['fake_B', 'real_B','fake_BT', 'real_BT']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            #self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            #self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr)
            self.optimizers.append(self.optimizer_G)
            self.criterionMSE = torch.nn.MSELoss(reduction='sum')
        else:
            print("----test data----")
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss(reduction='sum')

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device1)
        self.real_B = input['B' if AtoB else 'A'].to(self.device1)
        self.real_C = input['C'].to(self.device1)
        #self.real_D = input['D'].to(self.device1)  
        #self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self,epoch1,lstart):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        #netin1 = self.real_A[:, :, 1:800:2, :]
        [self.fake_B,self.grad] = self.netG(self.real_B,self.real_A,lstart,epoch1,self.real_B,self.real_B)  # G(A)
        #self.fake_B = torch.clamp(self.fake_B,min=15.00,max=35.50)
        #filen = './marmousi/Gr1ad' + str(131)+'ep'+str(epoch1)+'.npy' #switch on for physics based fwi       
        #np.save(filen, self.real_A.cpu().detach().numpy())  #switch on physics based fwi
        # print(np.shape(self.fake_B))
        # print(self.fake_B.get_device())

    def forwardT(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        False_lstart = 1
        False_epoch = -1
        #netin1 = self.real_A[:, :, 1:800:2, :]
        [self.fake_BT,self.gradT] = self.netG(self.real_B,self.real_A,False_lstart,False_epoch,self.real_B,self.real_B)  # G(A)
        #self.fake_BT = torch.clamp(self.fake_BT,min=15.00,max=35.50)
        self.real_BT = self.real_B

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        # Second, G(A) = B
        #print("real B shape")
        diff_size = self.real_B.size()
        self.loss_M_MSE = (self.criterionMSE(self.fake_B, self.real_B)) * \
            100/(diff_size[0]*diff_size[1]*diff_size[2]*diff_size[3])
        self.loss_D_MSE = 0.0
        # combine loss and calculate gradients
        self.loss_G = self.loss_M_MSE
        self.loss_G.backward()
    
    def backward_GKL(self,epoch1):
        """Calculate MSE loss along with KL divergence"""
                # First, G(A) should fake the discriminator
        # Second, G(A) = B
        #print("real B shape")
        diff_size = self.real_B.size()
        self.loss_M_MSE = (self.criterionMSE(self.fake_B, self.real_B)) * \
            100/(diff_size[0]*diff_size[1]*diff_size[2]*diff_size[3])
        self.loss_D_MSE = 0.0
        # combine loss and calculate gradients
        self.loss_G = self.loss_M_MSE
        self.loss_G.backward()
        
    def backward_G11(self, epoch1, batch, lstart):
            
        """Calculate GAN and L1 loss for the generator"""
        #lstart = 1
        #lstart2 = 50
        diff_size = self.real_B.size()

        #if (epoch1 > lstart):
        #    self.loss_M1_MSE = self.criterionMSE(self.fake_B, self.fake_BD)/(diff_size[0]*diff_size[1]*diff_size[2]*diff_size[3])
        #else:
        self.loss_M1_MSE = 0.0

        
        self.loss_D_MSE = 0.0
        #print("shape of real_C :", np.shape(self.real_C))
        #print("shape of fake_B :", np.shape(self.fake_B))
        #1000 is the best model for vae
        self.loss_M_MSE = self.criterionMSE(self.real_B, self.fake_B)/(diff_size[0]*diff_size[1]*diff_size[2]*diff_size[3])
        #k
        #kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0)
        #self.loss_K_MSE = kld_loss/diff_size[0]
        #self.loss_M_MSE = 0.0
        #self.loss_K_MSE = 0.0
        #print("loss_M_MSE : ",self.loss_M_MSE)
        
        #print("loss_K_MSE : ",self.loss_K_MSE)
        #print("mu :", self.mu)
        #print("shape of mu :", self.mu.size())
        #print("var :", self.log_var)
        #print("shape of var :", self.log_var.size())
        #print("loss MSE example :", self.loss_M_MSE)
        #print("diff size :", diff_size)
        #print("device of fake B:",str(self.fake_B.get_device()))
        
        ######filen = './marmousi/ZZ3M' + str(batch)+'ep'+str(epoch1)+'.npy'
        #######np.save(filen, self.z.cpu().detach().numpy()) 
        
        ####filen = './marmousi/ZZConstant' + str(batch)+'ep'+str(epoch1)+'.npy'
        ######np.save(filen, self.z.cpu().detach().numpy())
        
        ########filen = './marmousi/MT3M' + str(batch)+'ep'+str(epoch1)+'.npy'
        ##########np.save(filen, self.fake_B.cpu().detach().numpy()) 
        
        ####filen = './marmousi/FinalInvConstant' + str(batch)+'ep'+str(epoch1)+'.npy'
        ######np.save(filen, self.fake_B.cpu().detach().numpy()) 
        
        # if (epoch1 > lstart):
        #      filen = './deepwave/fake29Sep' + \
        #          str(batch)+'ep'+str(epoch1)+'.npy'
        #      np.save(filen, self.fake_B.cpu().detach().numpy())
        #      filen = './deepwave/realA29Sep' + \
        #          str(batch)+'ep'+str(epoch1)+'.npy'
        #      np.save(filen, self.real_A.cpu().detach().numpy())
        #      filen = './deepwave/realB29Sep' + \s
        #          str(batch)+'ep'+str(epoch1)+'.npy'
        #      np.save(filen, self.real_B.cpu().detach().numpy())
        #     filen = './deepwave/fakeData11Sep' + \
        #            str(batch)+'ep'+str(epoch1)+'.npy'
        #     np.save(filen, self.fake_BD.cpu().detach().numpy())

        lambda1 = 1
        lambda2 = 0
        if (epoch1>lstart):
            lambda1 = 0.5
            lambda2 = 0.5
            
        #self.fake_B.retain_grad()

        
        self.loss_G = lambda1 * self.loss_M_MSE + lambda2 * self.loss_M1_MSE
        #####self.loss_G = lambda2 * self.loss_M1_MSE
        self.loss_G.backward(retain_graph=True)
        ####self.loss_G.backward()
        
        #maxb = torch.max(torch.abs(self.fake_B.grad))
        
        #print("maxb :", maxb)
        
        if (epoch1>lstart):
            #maxg = torch.max(torch.abs(self.grad))
        
            #self.fake_B.grad = None
            self.grad = self.grad*(10**5)   #####(10**5) works for marmousi model
            #self.grad = tgm.image.gaussian_blur(self.grad, (5, 5), (10, 10))
            ##self.grad[:,:,0:26,:] = 0
            ###self.grad = scipy.ndimage.gaussian_filter(self.grad,10)
            #maxg = torch.max(torch.abs(self.grad))
            #print("maxg :", maxg)
        
        #self.fake_B.register_hook(print)
        #filen = './marmousi/GradNewAD' + str(batch)+'ep'+str(epoch1)+'.npy' #switch on for physics based fwi       
        #np.save(filen, self.fake_B.grad.cpu().detach().numpy())  #switch on physics based fwi
        #print("shape of fake B grad :", self.fake_B.grad)
        #grad = torch.unsqueeze(torch.unsqueeze(self.grad,0),1) #switch on for physics based fwi
        #grad = grad.to(self.fake_B.get_device()) #switch on for physics based fwi
        #print("shape of self grad :", np.shape(self.grad))
        
        #self.grad = self.grad/torch.max(self.grad.abs())
            self.fake_B.backward(self.grad) #switch on for physics based fwi
        
        
        #print("shape of fake_B :", np.shape(self.fake_B))
        #print("shape of grad :", np.shape(self.grad))   
        if (epoch1 % 20 == 0): 
            filen = './marmousi/GradAD' + str(batch)+'ep'+str(epoch1)+'.npy' #switch on for physics based fwi       
            np.save(filen, self.grad.cpu().detach().numpy())  #switch on physics based fwi
        
            filen = './marmousi/FakeAD' + str(batch)+'ep'+str(epoch1)+'.npy' #switch on for physics based fwi       
            np.save(filen, self.fake_B.cpu().detach().numpy())  #switch on physics based fwi
        
        #filen = './marmousi/RealAD' + str(batch)+'ep'+str(epoch1)+'.npy' #switch on for physics based fwi       
        #np.save(filen, self.real_B.cpu().detach().numpy())  #switch on physics based fwi
        #if (epoch1 == 52):
        #    np.save('true_data.npy',self.real_A.cpu().detach().numpy())
        #    np.save('true_model.npy',self.real_B.cpu().detach().numpy())


    def optimize_parameters(self, epoch, batch, lstart):
        self.forward(epoch,lstart)                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G11(epoch,batch,lstart)                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def compute_loss_only(self):
        #lossL1 = self.criterionL1(self.fake_BT,self.real_BT)
        #self.loss_V_L1 = lossL1
        #print("Loss L1 : "+ str(lossL1.cpu().float().numpy()))
        diff_size = self.real_B.size()
        lossMSE = self.criterionMSE(
            self.fake_BT, self.real_B)/(diff_size[0]*diff_size[1]*diff_size[2]*diff_size[3])
        self.loss_V_MSE = lossMSE
        print("Loss RMSE : "+str(lossMSE.cpu().float().numpy()))
        #lossSSIM = metrics.structural_similarity(np.squeeze(self.fake_B.cpu().float().numpy()),np.squeeze(self.real_B.cpu().float().numpy()) )
        #print("Loss SSIM :"+str(lossSSIM))
        #loss_RMSE = torch.nn.MSELoss(self.fake_B,self.real_B)
        #loss_RMSE = np.sqrt(loss_RMSE.cpu().float().numpy())
        #print("RMSE loss :" + loss_RMSE)
        pass

    def update_epoch(self, epoch):
        # update epoch number
        self.epoch1 = epoch
        print("epoch numbers : "+str(self.epoch1))



        
