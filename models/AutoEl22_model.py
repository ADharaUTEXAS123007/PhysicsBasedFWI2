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
from seisgan.optimizers import MALA, SGHMC
#from vgg import Vgg16

#from skimage import metrics


class AutoEl22Model(BaseModel):
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
        parser.set_defaults(norm='batch', netG='AutoEl22',
                            dataset_mode='unalignedVelABCDEl', ngf='32')
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
        #torch.manual_seed(5465)
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
        self.loss_names = ['D_MSE', 'M_MSE', 'V_MSE', 'K_MSE']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['fake_Vp','real_Vp','fake_Vs', 'real_Vs','fake_Rho','real_Rho']
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
            #self.optimizer_G = MALA(self.netG.parameters(), lr=opt.lr)
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
        self.real_D = input['D'].to(self.device1)  
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
    
    def forward2(self, epoch1, lstart, freq):
        if (epoch1 == 1):
            self.latent = torch.ones(1,1,1,1)
        [self.fake_B,self.grad,self.latent,self.vp_grad,self.vs_grad,self.rho_grad,self.loss_D_MSE] = self.netG(self.real_B,self.real_A,lstart,epoch1,self.latent,self.real_C,self.real_D,freq)  # G(A)
        self.real_Vp = torch.unsqueeze(self.real_B[:,0,:,:],1)
        self.real_Vs = torch.unsqueeze(self.real_B[:,1,:,:],1)
        self.real_Rho = torch.unsqueeze(self.real_B[:,2,:,:],1)
        
        self.low_Vp = torch.unsqueeze(self.real_C[:,0,:,:],1)
        self.low_Vs = torch.unsqueeze(self.real_C[:,1,:,:],1)
        self.low_Rho = torch.unsqueeze(self.real_C[:,2,:,:],1)
        
        self.fake_Vp = torch.unsqueeze(self.fake_B[:,0,:,:],1)
        self.fake_Vs = torch.unsqueeze(self.fake_B[:,1,:,:],1)
        self.fake_Rho = torch.unsqueeze(self.real_B[:,2,:,:],1)
        
        self.vp_grad = torch.unsqueeze(self.vp_grad,0)
        self.vs_grad = torch.unsqueeze(self.vs_grad,0)
        self.rho_grad = torch.unsqueeze(self.rho_grad,0)
        
        self.grad = torch.cat((self.vp_grad,self.vs_grad),dim=0)
        #self.grad = self.vp_grad
        self.grad = torch.unsqueeze(self.grad,0)
        return self.loss_D_MSE

    def forward(self,epoch1,lstart,freq):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        #netin1 = self.real_A[:, :, 1:800:2, :]
        if (epoch1 == 1):
            self.latent = torch.ones(1,1,1,1)
        [self.fake_Vp,self.fake_Vs,self.grad,self.latent,self.vp_grad,self.vs_grad,self.rho_grad,self.loss_D_MSE] = self.netG(self.real_B,self.real_A,lstart,epoch1,self.latent,self.real_C,self.real_D,freq)  # G(A)
        self.real_Vp = torch.unsqueeze(self.real_B[:,0,:,:],1)
        self.real_Vs = torch.unsqueeze(self.real_B[:,1,:,:],1)
        self.real_Rho = torch.unsqueeze(self.real_B[:,2,:,:],1)
        
        self.low_Vp = torch.unsqueeze(self.real_C[:,0,:,:],1)
        self.low_Vs = torch.unsqueeze(self.real_C[:,1,:,:],1)
        self.low_Rho = torch.unsqueeze(self.real_C[:,2,:,:],1)
        
        #self.fake_Vp = torch.unsqueeze(self.fake_B[:,0,:,:],1)
        #self.fake_Vs = torch.unsqueeze(self.fake_B[:,1,:,:],1)
        self.fake_B = torch.cat((self.fake_Vp,self.fake_Vs),axis=1)
        self.fake_Rho = torch.unsqueeze(self.real_B[:,2,:,:],1)
        
        self.vp_grad = torch.unsqueeze(self.vp_grad,0)
        self.vs_grad = torch.unsqueeze(self.vs_grad,0)
        self.rho_grad = torch.unsqueeze(self.rho_grad,0)
        
        self.grad = torch.cat((self.vp_grad,self.vs_grad),dim=0)
        #self.grad = self.vp_grad
        self.grad = torch.unsqueeze(self.grad,0)
        #self.latent = self.latent.clone().detach()
        #print("self.latent :", self.latent)
        #self.real_C = self.fake_B
        #self.real_B = self.real_C
        #self.fake_B = torch.clamp(self.fake_B,min=15.00,max=35.50)
        #filen = './marmousi/Gr1ad' + str(131)+'ep'+str(epoch1)+'.npy' #switch on for physics based fwi       
        #np.save(filen, self.real_A.cpu().detach().numpy())  #switch on physics based fwi
        # print(np.shape(self.fake_B))
        # print(self.fake_B.get_device())
           

    def forwardT(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        False_lstart = 1
        False_epoch = -1
        freq = 3
        #netin1 = self.real_A[:, :, 1:800:2, :]
        #if (epoch1 == 1):
        self.latentT = torch.ones(1,1,1,1)
        [self.VpT,self.VsT,self.gradT,self.latentT,_,_,_,_] = self.netG(self.real_B,self.real_A,False_lstart,False_epoch,self.latentT,self.real_C,self.real_D,freq)  # G(A)
        #self.fake_BT = torch.clamp(self.fake_BT,min=15.00,max=35.50)
        #self.real_VpT = self.real_B[:,0,:,:]
        #self.real_VsT = self.real_B[:,1,:,:]
        #self.real_RhoT = self.real_B[:,2,:,:]
        self.fake_BT = torch.cat((self.VpT,self.VsT),axis=1)
        #self.fake_VpT = self.fake_BT[:,0,:,:]
        #self.fake_VsT = self.fake_BT[:,1,:,:]
        #self.fake_RhoT = self.fake_BT[:,2,:,:]
        
        #self.real_C = self.real_BT
        

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
        #self.loss_D_MSE = 0.0
        
        #self.loss_D_MSE = 0.0
        self.loss_K_MSE = 0.0
        #print("shape of real_C :", np.shape(self.real_C))
        #print("shape of fake_B :", np.shape(self.fake_B))
        #1000 is the best model for vae
        #print("shape of real_B :", np.shape(self.real_B))
        #print("shape of fake_B :", np.shape(self.fake_B))
        #print("D_MSE :", self.loss_D_MSE)
        
        #self.loss_M_MSE = self.criterionMSE(self.real_B, self.fake_B)/(diff_size[0]*diff_size[1]*diff_size[2]*diff_size[3])
        self.loss_M_MSE = self.criterionMSE(self.low_Vp, self.fake_Vp) + self.criterionMSE(self.low_Vs, self.fake_Vs) + self.criterionMSE(self.low_Rho, self.fake_Rho)
        
        print("loss M MSE :", self.loss_M_MSE)
        print("loss D MSE :", self.loss_D_MSE)
        
        #print("shape of grad :", np.shape(self.grad))
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
        #if (epoch1>lstart):
        #    lambda1 = 0.5
        #    lambda2 = 0.5
            
        #self.fake_B.retain_grad()

        
        self.loss_G = lambda1 * self.loss_M_MSE 
        ####self.loss_G = lambda2 * self.loss_M1_MSE
        
        #if (epoch1 <= lstart):
            #print("1st epoch1 :", epoch1)
            #self.loss_G.backward()
        self.loss_G.backward()
        
        #maxb = torch.max(torch.abs(self.fake_B.grad))
        
        #print("maxb :", maxb)
        lstart1 = 35
        lstart2 = 60
        
        if (epoch1>lstart):
            print("2nd epoch1 :", epoch1)
            #self.loss_G.backward(retain_graph=True)
            #self.optimizer_G.zero_grad()
            #maxb = torch.max(torch.abs(self.fake_B.grad))
            #maxg = torch.max(torch.abs(self.grad))
        
            #self.fake_B.grad = None
            #self.fake_B.grad = None
            #if (epoch1>lstart and epoch1<=lstart1):
            #self.grad = self.grad*(10**5)  #####(10**5) works for marmousi model
            #self.grad = torch.clip(self.grad, min=-0.1, max=0.1)
                
            #if (epoch1>lstart1 and epoch1<=lstart2):
            #    self.grad = self.grad*(10**6)  #####(10**5) works for marmousi model
            #    self.grad = torch.clip(self.grad, min=-0.1, max=0.1)
                
            #if (epoch1>lstart2):
            #    self.grad = self.grad*(10**7)   #####(10**5) works for marmousi model
            #    self.grad = torch.clip(self.grad, min=-2.0, max=2.0)
            #self.grad = (self.grad-1600)/(2300-1600)
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
            self.grad = self.grad.cuda(self.fake_B.get_device())
            print("gradient device :",self.grad.get_device())
            
            print("gradient shape :", np.shape(self.grad))
            
            self.fake_B.backward(self.grad) #switch on for physics based fwi
        
        
        #print("shape of fake_B :", np.shape(self.fake_B))
        #print("shape of grad :", np.shape(self.grad))   
        if (epoch1 % 1 == 0): 
            #filen = './marmousi22/Grad20Dec2AD' + str(batch)+'ep'+str(epoch1)+'.npy' #switch on for physics based fwi       
            #np.save(filen, self.grad.cpu().detach().numpy())  #switch on physics based fwi
            
            filen = './marmousiEl/Grad2JanAD' + str(batch)+'ep'+str(epoch1)+'.npy' #switch on for physics based fwi       
            np.save(filen, self.grad.cpu().detach().numpy())  #switch on physics based fwi
        
            #filen = './marmousi22/Fake20Dec2AD' + str(batch)+'ep'+str(epoch1)+'.npy' #switch on for physics based fwi       
            #np.save(filen, self.fake_B.cpu().detach().numpy())  #switch on physics based fwi
            
            filen = './marmousiEl/Fake2JanAD' + str(batch)+'ep'+str(epoch1)+'.npy' #switch on for physics based fwi       
            np.save(filen, self.fake_B.cpu().detach().numpy())  #switch on physics based fwi
            
            #filen = './marmousi22/Real20Dec2AD' + str(batch)+'ep'+str(epoch1)+'.npy' #switch on for physics based fwi       
            #np.save(filen, self.real_B.cpu().det2ach().numpy())  #switch on physics based fwi
            
            filen = './marmousiEl/Real2JanAD' + str(batch)+'ep'+str(epoch1)+'.npy' #switch on for physics based fwi       
            np.save(filen, self.real_B.cpu().detach().numpy())  #switch on physics based fwi
        
        #filen = './marmousi/RealAD' + str(batch)+'ep'+str(epoch1)+'.npy' #switch on for physics based fwi       
        #np.save(filen, self.real_B.cpu().detach().numpy())  #switch on physics based fwi
        #if (epoch1 == 52):
        #    np.save('true_data.npy',self.real_A.cpu().detach().numpy())
        #    np.save('true_model.npy',self.real_B.cpu().detach().numpy())

        
    def closure(self,epoch,lstart,batch,freq):
        #epoch = 1
        #lstart = 0
        #batch = 0
        loss = self.forward2(epoch, lstart, freq)
        loss = loss*10**20
        loss = np.array(loss)
        loss = loss.astype(float)
        self.optimizer_G.zero_grad()
        self.backward_G11(epoch,batch,lstart)
        print("loss type")
        #losst.backward()
        #loss1 = np.empty([1,1])
        #loss1[0,0] = loss
        loss1 = torch.from_numpy(loss)
        loss1 = loss1.float()
        #loss1 = loss1.float()
        #loss.item = 1
        #loss = torch.unsqueeze(loss,1)
        #print("shape of loss :", loss1)
        print("loss1 :", loss1)
        return loss1


    def optimize_parameters(self, epoch, batch, lstart,freq):
        self.forward(epoch,lstart,freq)                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G11(epoch,batch,lstart)   
        #                 # calculate graidents for G
        
        ######self.optimizer_G.step(lambda : self.closure(epoch, lstart, batch, freq))             # udpate G's weights
        self.optimizer_G.step()

    def compute_loss_only(self):
        #lossL1 = self.criterionL1(self.fake_BT,self.real_BT)
        #self.loss_V_L1 = lossL1
        #print("Loss L1 : "+ str(lossL1.cpu().float().numpy()))
        diff_size = self.real_B.size()
        lossMSE = self.criterionMSE(
            self.fake_BT, self.real_B[:,0:2,:,:])/(diff_size[0]*diff_size[1]*diff_size[2]*diff_size[3])
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



        
