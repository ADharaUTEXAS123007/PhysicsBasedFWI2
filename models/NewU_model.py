import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import sys
#sys.path.append('../deepwave')
#import deepwave

#from skimage import metrics


class NewUModel(BaseModel):
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
        parser.set_defaults(norm='batch', netG='NewU', dataset_mode='unalignedVel2', ngf='32')
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
        torch.cuda.set_device(0)
        torch.cuda.set_device(1)
        self.device1 = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu') 
        #self.device2 = torch.device('cuda:{}'.format(self.gpu_ids[1])) if self.gpu_ids else torch.device('cpu') 
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_MSE','M_MSE','V_MSE']
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
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr)
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
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        netin1 = self.real_A[:,:,1:2000:5,:]
        self.fake_B = self.netG(netin1)  # G(A)
        #print(np.shape(self.fake_B))
        #print(self.fake_B.get_device())


    def forwardT(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        netin1 = self.real_A[:,:,1:2000:5,:]
        self.fake_BT = self.netG(netin1)  # G(A)
        self.real_BT = self.real_B

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        # Second, G(A) = B
        #print("real B shape")
        diff_size = self.real_B.size()
        self.loss_M_MSE = (self.criterionMSE(self.fake_B, self.real_B))*100/(diff_size[0]*diff_size[1]*diff_size[2]*diff_size[3])
        self.loss_D_MSE = 0.0
        # combine loss and calculate gradients
        self.loss_G = self.loss_M_MSE
        self.loss_G.backward()

    def backward_G1(self,epoch1):
        """Calculate GAN and L1 loss for the generator"""
        # calculate the source amplitudes
        # GPU_string='cuda:0'
        # device1 = torch.device(GPU_string)
        # GPU_string='cuda:1'
        # device2 = torch.device(GPU_string)

        #---------deepwave------------#
        t = epoch1
        freq = 25
        dx = 10
        nt = 2000
        dt = 0.001
        num_shots = 15
        num_receivers_per_shot = 301
        num_sources_per_shot = 1
        num_dims = 2
        #ModelDim = [201,301]
        source_spacing = 301 * dx / num_shots
        receiver_spacing = 301 * dx / num_receivers_per_shot
        x_s = torch.zeros(num_shots, num_sources_per_shot, num_dims)
        x_s[:, 0, 1] = torch.arange(num_shots).float() * source_spacing
        x_r = torch.zeros(num_shots, num_receivers_per_shot, num_dims)
        x_r[0, :, 1] = torch.arange(num_receivers_per_shot).float() * receiver_spacing
        x_r[:, :, 1] = x_r[0, :, 1].repeat(num_shots, 1)

        source_amplitudes_true = (deepwave.wavelets.ricker(freq, nt, dt, 1/freq)
                                        .reshape(-1, 1, 1))
        source_amplitudes_true = source_amplitudes_true.to(self.device2)
        lstart = -1
        num_batches = 3
        num_epochs = 1
        if (t > lstart):
         num_epochs = 5
        num_shots_per_batch = int(num_shots / num_batches)
        diff_size = self.real_B.size()
        #print("size of self.realA")
        #print(np.shape(self.real_A))
        sumlossinner = 0.0

        ################data misfit calculation##########################################

        for k in range(diff_size[0]):
            net1out = self.fake_B[k,0,:,:]
            receiver_amplitudes_true = self.real_A[k,:,:,:]
            receiver_amplitudes_true = receiver_amplitudes_true.swapaxes(0,1)
            rcv_amps_true_max, _ = receiver_amplitudes_true.max(dim=0, keepdim=True)
            rcv_amps_true_norm = receiver_amplitudes_true / (rcv_amps_true_max.abs() + 1e-10)
            #print("receiver amplitude true shape")
            #print(np.shape(receiver_amplitudes_true))
            net1out1 = net1out.detach()
                #net1out1 = torch.tensor(net1out1)
            net1out1 = net1out1*(4500-2000)+2000
                #np.save('ftout1',net1out1.cpu().numpy())
            net1out1 = net1out1.to(self.device2)
            net1out1.requires_grad = True
            criterion = torch.nn.MSELoss()
            optimizer2 = torch.optim.Adam([{'params': [net1out1], 'lr':10}])

            for epoch in range(num_epochs):
                for it in range(num_batches):
                    optimizer2.zero_grad()
                    prop = deepwave.scalar.Propagator({'vp':net1out1},dx)
                    batch_src_amps = source_amplitudes_true.repeat(1, num_shots_per_batch, 1)
                    #print("shape of batch src amps")
                    #print(np.shape(batch_src_amps))
                    batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches].to(self.device2)
                    batch_x_s = x_s[it::num_batches].to(self.device2)
                    #print("shape of batch src amps")
                    #print(np.shape(batch_x_s))
                    batch_x_r = x_r[it::num_batches].to(self.device2)
                    #print("shape of batch receiver amps")
                    #print(np.shape(batch_x_r))
                    batch_rcv_amps_pred = prop(batch_src_amps, batch_x_s, batch_x_r, dt)
                    batch_rcv_amps_pred_max, _ = batch_rcv_amps_pred.max(dim=0, keepdim=True)
                    # Normalize amplitudes by dividing by the maximum amplitude of each receiver
                    batch_rcv_amps_pred_norm = batch_rcv_amps_pred / (batch_rcv_amps_pred_max.abs() + 1e-10)
                    #print("shape of receiver amplitudes predicted")
                    #print(np.shape(batch_rcv_amps_pred))
                    lossinner = criterion(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
                    if (epoch == num_epochs-1):
                        sumlossinner += lossinner.item()
                    if (t > lstart):
                        lossinner.backward()
                        #epoch_loss += loss.item()
                        optimizer2.step()

            net1out1 = (net1out1-2000)/(4500-2000)
            net1out1 = torch.unsqueeze(net1out1,0)
            if k==0:
                data1outs = net1out1
            else:
                data1outs = torch.cat((data1outs,net1out1),0)
        
        data1outs = data1outs.to(self.device1)
        data1outs = torch.unsqueeze(data1outs,1)

        #-------------deepwave---------------------#

        #print("shape of data1outs")
        #print(np.shape(data1outs))

        #print("shape of real B")
        #print(np.shape(self.real_B))
        self.loss_D_MSE = sumlossinner/diff_size[0]
        loss_data = (self.criterionMSE(self.fake_B, data1outs))/(diff_size[0]*diff_size[1]*diff_size[2]*diff_size[3])
        print("----loss_data-----")
        print(loss_data)
        self.loss_M_MSE = (self.criterionMSE(self.fake_B, self.real_B))/(diff_size[0]*diff_size[1]*diff_size[2]*diff_size[3])
        print("---loss MSE-----")
        print(self.loss_M_MSE)

        lambda1 = 0
        lambda2 = 1
        if (t>lstart):
            lambda2 = 1
        if (t>100):
            lambda1 = 0
        #print("D_MSE loss")
        #print(self.loss_D_MSE)

        #print("M_MSE loss")
        #print(self.loss_M_MSE)

        #print("loss data")
        #print(loss_data)
        # combine loss and calculate gradients
        self.loss_G =  lambda1*self.loss_M_MSE*100 + lambda2*loss_data*100
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def compute_loss_only(self):
        #lossL1 = self.criterionL1(self.fake_BT,self.real_BT)
        #self.loss_V_L1 = lossL1
        #print("Loss L1 : "+ str(lossL1.cpu().float().numpy()))
        diff_size = self.real_B.size()
        lossMSE = self.criterionMSE(self.fake_BT,self.real_BT)/(diff_size[0]*diff_size[1]*diff_size[2]*diff_size[3])
        self.loss_V_MSE = lossMSE
        print("Loss RMSE : "+str(lossMSE.cpu().float().numpy()))
        #lossSSIM = metrics.structural_similarity(np.squeeze(self.fake_B.cpu().float().numpy()),np.squeeze(self.real_B.cpu().float().numpy()) )
        #print("Loss SSIM :"+str(lossSSIM))
        #loss_RMSE = torch.nn.MSELoss(self.fake_B,self.real_B)
        #loss_RMSE = np.sqrt(loss_RMSE.cpu().float().numpy())
        #print("RMSE loss :" + loss_RMSE)
        pass

    def update_epoch(self,epoch):
        #update epoch number
        self.epoch1 = epoch
        print("epoch numbers : "+str(self.epoch1))


