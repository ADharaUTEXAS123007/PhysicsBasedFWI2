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
import ray
import time
import os

#from skimage import metrics


class VaeModel(BaseModel):
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
        parser.set_defaults(norm='batch', netG='Vae',
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
        #print("number of cuda devices:", torch.cuda.device_count())
        #for i in range(3):
        #torch.cuda.set_device(1)

        #torch.cuda.set_device(2)
        #del self.device
        # print(self.device)
        # Start Ray.
        #os.environ['CUDA_VISIBLE_DEVICES'] = "2,3,4,5,6,7"
        #ray.init(num_cpus=48,num_gpus=6)

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
        self.visual_names = ['fake_BT', 'real_BT']
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
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self,epoch1,lstart):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        #netin1 = self.real_A[:, :, 1:800:2, :]
        #lstart = 1
        [self.fake_B, self.mu, self.log_var, self.grad] = self.netG(self.real_B,self.real_A,lstart,epoch1,self.real_B,self.real_C)  # G(A)
        # print(np.shape(self.fake_B))
        # print(self.fake_B.get_device())

    def forwardT(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        #netin1 = self.real_A[:, :, 1:800:2, :]
        False_lstart = 1
        False_epoch = -1
        #netin1 = self.real_A[:, :, 1:800:2, :]
        [self.fake_BT, self.muT, self.log_varT, self.gradT] = self.netG(self.real_B,self.real_A,False_lstart,False_epoch,self.real_B,self.real_C)  # G(A)
        #self.fake_BT = torch.clamp(self.fake_BT,min=15.00,max=35.50)
        self.real_BT = self.real_B
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
    
    def backward_GKL(self, epoch1):
        """Calculate MSE loss along with KL divergence"""
                # First, G(A) should fake the discriminator
        # Second, G(A) = B
        #print("real B shape")
        diff_size = self.real_B.size()
        self.loss_M_MSE = (self.criterionMSE(self.fake_B, self.real_B)) * \
            100/(diff_size[0]*diff_size[1]*diff_size[2]*diff_size[3])
        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0)
        self.loss_K_MSE = kld_loss
        #self.loss_K_MSE = 0.0

        #print("KL divergence loss :", kld_loss)
        #print("MSE loss :", self.loss_M_MSE)
        self.loss_D_MSE = 0.0
        self.loss_M1_MSE = 0.0
        # combine loss and calculate gradients
        self.loss_G = self.loss_M_MSE + self.loss_K_MSE
        self.loss_G.backward()
        #if (epoch1 == 195):
        #    np.save('real.npy',self.real_B)
        #    np.save('fake.npy',self.fake_B)
        #    np.save('real_seismic.npy',self.real_A)
        #    np.save('mu.npy',self.mu)
        #    np.save('var.npy',self.log_var)


    def backward_G1(self, epoch1, batch, lstart):
        
        """Calculate GAN and L1 loss for the generator"""
        #lstart = 4
        lstart2 = 50
        diff_size = self.real_B.size()

        if (epoch1 > lstart):
            

            result_ids1 = []
            result_ids2 = []
            if (epoch1 > lstart):
               filen = './deepwave/batchOld' + \
                   str(batch)+'ep'+str(epoch1)+'.npy'
               np.save(filen, self.fake_B.cpu().detach().numpy())
               filen = './deepwave/realOld' + \
                   str(batch)+'ep'+str(epoch1)+'.npy'
               np.save(filen, self.real_B.cpu().detach().numpy())

            for k in range(diff_size[0]):
                po = self.prop.remote(self, epoch1, k, lstart)
                result_ids1.append(po[0])
                result_ids2.append(po[1])
            
            #print("result ids2 :", str(result_ids2))

            # #-------------deepwave---------------------#
            lossinner = ray.get(result_ids2)
            data1outs = ray.get(result_ids1)
            lossinner = np.expand_dims(lossinner, axis=1)

            #print("shape of lossinner")
            #print(np.shape(lossinner))
            #print("diff_size[0] :",str(diff_size[0]))
            data1outs = np.array(data1outs)
            #print("shape of data1outs")
            #print(np.shape(data1outs))
            if (epoch1 > lstart):
               filen = './deepwave/batch75New' + \
                   str(batch)+'ep'+str(epoch1)+'.npy'
               np.save(filen, data1outs)
            data1outs = torch.from_numpy(data1outs)
            data1outs = data1outs.to(self.device1)
            data1outs = torch.unsqueeze(data1outs, 1)

            self.loss_D_MSE = np.mean(lossinner)
            self.loss_M1_MSE = self.criterionMSE(self.fake_B, data1outs)*100/(diff_size[0]*diff_size[1]*diff_size[2]*diff_size[3])
        else:
            loss_data = 0.0
            self.loss_D_MSE = 0.0
            self.loss_M1_MSE = 0.0
        self.loss_M_MSE = self.criterionMSE(self.fake_B, self.real_B)*100/(diff_size[0]*diff_size[1]*diff_size[2]*diff_size[3])
        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0)
        self.loss_K_MSE = kld_loss
        #print("loss MSE example :", self.loss_M_MSE)
        #print("diff size :", diff_size)
        

        lambda1 = 1
        lambda2 = 0
        if (epoch1>lstart):
            lambda1 = 0.7
            lambda2 = 0.3
            

        self.loss_G = lambda1 * self.loss_M_MSE + self.loss_K_MSE + lambda2 * self.loss_M1_MSE
        #self.loss_G = lambda2 * self.loss_M1_MSE
        self.loss_G.backward()
        #if (epoch1 == 52):
        #    np.save('true_data.npy',self.real_A.cpu().detach().numpy())
        #    np.save('true_model.npy',self.real_B.cpu().detach().numpy())

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
        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0)
        self.loss_K_MSE = kld_loss/(diff_size[0])
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

        lambda1 = 10
        lambda2 = 0
        if (epoch1>lstart):
            lambda1 = 0.5
            lambda2 = 0.5
            
        #self.fake_B.retain_grad()

        
        self.loss_G = lambda1 * self.loss_M_MSE + self.loss_K_MSE 
        ####self.loss_G = lambda2 * self.loss_M1_MSE
        ##self.loss_G.backward(retain_graph=True)
        self.loss_G.backward()
        
        #maxb = torch.max(torch.abs(self.fake_B.grad))
        
        #print("maxb :", maxb)
        
        if (epoch1>lstart):
            #maxg = torch.max(torch.abs(self.grad))
        
            #self.fake_B.grad = None
            self.grad = self.grad*(10**7)   #####(10**5) works for marmousi model
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
            self.fake_B.backward(self.grad) #switch on for physics based fwi
        
        
        #print("shape of fake_B :", np.shape(self.fake_B))
        #print("shape of grad :", np.shape(self.grad))   
        if (epoch1 % 1 == 0): 
            filen = './marmousi/GradAD' + str(batch)+'ep'+str(epoch1)+'.npy' #switch on for physics based fwi       
            np.save(filen, self.grad.cpu().detach().numpy())  #switch on physics based fwi
        
            filen = './marmousi/FakeAD' + str(batch)+'ep'+str(epoch1)+'.npy' #switch on for physics based fwi       
            np.save(filen, self.fake_B.cpu().detach().numpy())  #switch on physics based fwi
            
            filen = './marmousi/RealAD' + str(batch)+'ep'+str(epoch1)+'.npy' #switch on for physics based fwi       
            np.save(filen, self.real_B.cpu().detach().numpy())  #switch on physics based fwi
        
        #filen = './marmousi/RealAD' + str(batch)+'ep'+str(epoch1)+'.npy' #switch on for physics based fwi       
        #np.save(filen, self.real_B.cpu().detach().numpy())  #switch on physics based fwi
        #if (epoch1 == 52):
        #    np.save('true_data.npy',self.real_A.cpu().detach().numpy())
        #    np.save('true_model.npy',self.real_B.cpu().detach().numpy())

    def optimize_parameters(self, epoch, batch, lstart):
        #lstart = 1
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
            self.fake_BT, self.real_BT)*10/(diff_size[0]*diff_size[1]*diff_size[2]*diff_size[3])
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


    @ray.remote(num_gpus=1,num_returns=2)
    def prop(self, epoch1, k, lstart):
        #---------deepwave------------#
        
        net1out1 = self.fake_B[k, 0, :, :]
        #print("k k :", str(k))
        net1out1 = net1out1.detach()
        #print(" ray gpu ids")
        g1 = ray.get_gpu_ids()[0]
        g1 = int(g1)
        #print("g1 :",g1)
        #torch.cuda.set_device(g1)
        self.devicek = torch.device('cuda')
        #     self.gpu_ids[7])) if self.gpu_ids else torch.device('cpu')
        #torch.cuda.set_device(int(g1))
        #if (g1 == 0):
            #torch.cuda.set_device(0)
        #    self.devicek = self.device1            
        # if (g1 == 1):
        #     torch.cuda.set_device(1)
        #     self.devicek = self.device2
        # if (g1 == 2):
        #     #torch.cuda.set_device(3)
        #     self.devicek = self.device3
        # if (g1 == 3):
        #     #torch.cuda.set_device(4)
        #     self.devicek = self.device4
        # if (g1 == 4):
        #     #torch.cuda.set_device(5)
        #     self.devicek = self.device5
        # if (g1 == 5):
        #     #torch.cuda.set_device(6)
        #     self.devicek = self.device6
        # if (g1 == 6):
        #     #torch.cuda.set_device(7)
        #     self.devicek = self.device7
        # if (g1 == 7):
        #     self.devicek = self.device8
        #t = epoch1
        freq = 15
        dx = 10
        nt = 800
        dt = 0.0015
        num_shots = 10
        num_receivers_per_shot = 101
        num_sources_per_shot = 1
        num_dims = 2
        #ModelDim = [201,301]
        source_spacing = 101 * dx / num_shots
        receiver_spacing = 101 * dx / num_receivers_per_shot
        x_s = torch.zeros(num_shots, num_sources_per_shot, num_dims)
        x_s[:, 0, 1] = torch.arange(num_shots).float() * source_spacing
        x_r = torch.zeros(num_shots, num_receivers_per_shot, num_dims)
        x_r[0, :, 1] = torch.arange(
            num_receivers_per_shot).float() * receiver_spacing
        x_r[:, :, 1] = x_r[0, :, 1].repeat(num_shots, 1)

        source_amplitudes_true = (deepwave.wavelets.ricker(freq, nt, dt, 1/freq)
                                  .reshape(-1, 1, 1))
        #print("device ordinal :", self.devicek)
        source_amplitudes_true = source_amplitudes_true.to(self.devicek)
        #lstart = -1
        num_batches = 1
        num_epochs = 1
        if (epoch1 > lstart):
            num_epochs = 1
        #if (epoch1 > 50):
        #    num_epochs = 30
        #if (epoch1 > 80):
        #    num_epochs = 40 
        num_shots_per_batch = int(num_shots / num_batches)
        #print("size of self.realA")
        # print(np.shape(self.real_A))
        sumlossinner = 0.0

        ################data misfit calculation##########################################

        #net1out1 = net1out1.to(self.devicek)

        receiver_amplitudes_true = self.real_A[k,:,:,:]
        receiver_amplitudes_true = receiver_amplitudes_true.swapaxes(0,1)
        #########rcv_amps_true_max, _ = torch.abs(receiver_amplitudes_true).max(dim=0, keepdim=True)
        ##########rcv_amps_true_norm = receiver_amplitudes_true / (rcv_amps_true_max.abs() + 1e-10)
        rcv_amps_true_norm = receiver_amplitudes_true
        #print("receiver amplitude true shape")
        # print(np.shape(receiver_amplitudes_true))
        #net1out1 = net1out.detach()
        #net1out1 = torch.tensor(net1out1)
        #net1out1 = net1out1*(4500-2000)+2000
        net1out1 = net1out1 * 100
        #min1 = torch.min(net1out1)
        #print(min1.get_device())
        #min1 = min1.to(self.device1)
        #mat2 = torch.ones(net1out1.size()[0],net1out1.size()[1]).to(self.device1)
        #mat2 = mat2 * min1
        #min1 = torch.min(net1out1)
        #max1 = torch.max(net1out1)
        #if (epoch1 == 52): 
        #np.save('./deepwave/before1.npy',net1out1.cpu().detach().numpy())
        # np.save('ftout1',net1out1.cpu().numpy())
        net1out1 = net1out1.to(self.devicek)
        #mat2 = mat2.to(self.devicek)
        #src_amps = source_amplitudes_true.repeat(
        #                1, num_shots, 1)
        #prop2 = deepwave.scalar.Propagator({'vp': mat2}, dx)
        #receiver_amplitudes_cte = prop2(src_amps,
        #                        x_s.to(self.devicek),
        #                        x_r.to(self.devicek), dt)

        criterion = torch.nn.MSELoss()

        #print("shape of mat2 :", np.shape(mat2))


        if (epoch1 > lstart):
            net1out1.requires_grad = True
            optimizer2 = torch.optim.Adam([{'params': [net1out1], 'lr':10}])

        for epoch in range(num_epochs):
                for it in range(num_batches):
                    if (epoch1 > lstart):
                        optimizer2.zero_grad()
                    model2 = net1out1.clone()
                    model2 = torch.clamp(net1out1,min=2000,max=4500)
                    #np.save('before108.npy',net1out1.cpu().detach().numpy())
                    #net1out1 = torch.clamp(net1out1,min=2000,max=4500)
                    prop = deepwave.scalar.Propagator({'vp': model2}, dx)
                    batch_src_amps = source_amplitudes_true.repeat(
                        1, num_shots_per_batch, 1)
                    #print("shape of batch src amps")
                    # print(np.shape(batch_src_amps))
                    batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches].to(self.devicek)
                    batch_x_s = x_s[it::num_batches].to(self.devicek)
                    #print("shape of batch src amps")
                    # print(np.shape(batch_x_s))
                    batch_x_r = x_r[it::num_batches].to(self.devicek)
                    #print("shape of batch receiver amps")
                    # print(np.shape(batch_x_r))
                    batch_rcv_amps_pred = 10 * prop(
                        batch_src_amps, batch_x_s, batch_x_r, dt)
                    #batch_rcv_amps_cte = receiver_amplitudes_cte[:,it::num_batches].to(self.devicek)
                    #batch_rcv_amps_pred = batch_rcv_amps_pred
                    ############batch_rcv_amps_pred_max, _ = torch.abs(batch_rcv_amps_pred).max(dim=0, keepdim=True)
                    # Normalize amplitudes by dividing by the maximum amplitude of each receiver
                    ##############batch_rcv_amps_pred_norm = batch_rcv_amps_pred / (batch_rcv_amps_pred_max.abs() + 1e-10)
                    batch_rcv_amps_pred_norm = batch_rcv_amps_pred
                    #print("shape of receiver amplitudes predicted")
                    # print(np.shape(batch_rcv_amps_pred))
                    lossinner = criterion(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
                    #filen = './deepwave/epoch1'+str(epoch)+'.npy'
                    #np.save(filen,net1out1.cpu().detach().numpy())
                    if (epoch == num_epochs-1):
                        sumlossinner += lossinner.item()
                    if (epoch1 > lstart):
                        lossinner.backward()
                        optimizer2.step()
                    #epoch_loss += loss.item()
                    #optimizer2.step()
        #if (epoch1 == 52): 
        #np.save('./deepwave/after1.npy',net1out1.cpu().detach().numpy())
        #np.save('./deepwave/seis231.npy',batch_rcv_amps_pred.cpu().detach().numpy())
        #net1out1 = (net1out1 - 2000)/(4500-2000)
        net1out1 = net1out1/100
        
    
        return net1out1.cpu().detach().numpy(),sumlossinner
        #return sumlossinner

    # @ray.remote (num_gpus=1)
    # def smallfun(self,epoch1,k):
    #     print(" ray gpu ids ")
    #     print(ray.get_gpu_ids()[0])
    #     self.devicek = torch.device('cuda')
    #     print("devicek")
    #     print(self.devicek)
    #     time.sleep(1)
    #     return(k)
        
