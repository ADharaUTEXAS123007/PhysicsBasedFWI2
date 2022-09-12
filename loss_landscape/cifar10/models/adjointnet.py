import torch
import torch.nn as nn
import numpy as np
import deepwave

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()
        # Kernel size: 3*3, Stride: 1, Padding: 1
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.LeakyReLU(0.1))
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.LeakyReLU(0.1))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True))
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs
    
class unetConv5(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv5, self).__init__()
        # Kernel size: 3*3, Stride: 1, Padding: 1
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.LeakyReLU(0.1))
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.LeakyReLU(0.1))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.LeakyReLU(0.1))
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.LeakyReLU(0.1))
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class unetDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm)
        self.down = nn.AvgPool2d(2, 2, ceil_mode=True)
        #self.dropout = nn.Dropout2d(0.1)
        

    def forward(self, inputs):
        outputs = self.conv(inputs)
        #outputs = self.bn(outputs)
        #outputs = self.lr(outputs)
        outputs = self.down(outputs)
        return outputs
    
class autoUp5(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm=True):
        super(autoUp5, self).__init__()
        self.conv = unetConv5(in_size, out_size, is_batchnorm)
        self.conv2 = unetConv5(out_size, out_size, is_batchnorm)
        # Transposed convolution
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, in_size, kernel_size=2,stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs2):
        outputs2 = self.up(inputs2)
        outputs3 = self.conv(outputs2)
        #offset1 = (outputs2.size()[2]-inputs1.size()[2])
        #offset2 = (outputs2.size()[3]-inputs1.size()[3])
        #padding=[offset2//2,(offset2+1)//2,offset1//2,(offset1+1)//2]
        # Skip and concatenate 
        #outputs1 = F.pad(inputs1, padding)
        return outputs3

class ADJOINTNET(nn.Module):
    def __init__(self,outer_nc=39, inner_nc=1, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ADJOINTNET, self).__init__()
        self.is_deconv     = False
        self.in_channels   = outer_nc
        self.is_batchnorm  = True
        self.n_classes     = 1
        
        #filters = [16, 32, 64, 128, 256]
        #filters = [32, 64, 128, 256, 512]
        #filters = [16, 32, 64, 128, 512]
        #######filters = [2, 4, 8, 16, 32] #this works best result so far for marmousi model
        #filters = [1, 1, 2, 4, 16]
        filters = [8, 16, 32, 64, 128] 
        #filters = [2, 4, 8, 16, 32]
        #filters = [1, 2, 4, 8, 16]
        #filters = [16, 32, 64, 128, 256]
        #########filters = [2, 4, 8, 16, 32]
        #filters = [32, 64, 128, 256, 512]
        
        latent_dim = 8
        self.combine1 = nn.Conv2d(self.in_channels, 1, 3, 1, 1)
        self.combine2 = nn.Conv2d(self.in_channels, 1, 3, 1, 1)
        
        self.down1   = unetDown(2, filters[0], self.is_batchnorm)
        #self.dropD1   = nn.Dropout2d(0.025)
        self.down2   = unetDown(filters[0], filters[1], self.is_batchnorm)
        #self.dropD2   = nn.Dropout2d(0.025)
        self.down3   = unetDown(filters[1], filters[2], self.is_batchnorm)
        #self.dropD3   = nn.Dropout2d(0.025)
        self.down4   = unetDown(filters[2], filters[3], self.is_batchnorm)
        #self.dropD4  = nn.Dropout2d(0.025)
        # self.center  = unetConv2(filters[3], filters[4], self.is_batchnorm)
        ##self.decoder_input1 = nn.Linear(filters[1]*250*51, latent_dim) #for marmousi 151x200
        #self.decoder_input1 = nn.Linear(filters[2]*125*26, latent_dim) #for marmousi 151x200
        #self.decoder_input = nn.Linear(latent_dim, filters[2]*500*102) #for marmousi 151x200
        self.decoder_input1 = nn.Linear(filters[3]*63*20, latent_dim) #for marmousi 101x101
        #self.decoder_input = nn.Linear(latent_dim, filters[3]*100*26) #for marmousi 101x101
        #self.decoder_input1 = nn.Linear(filters[1]*100*18, latent_dim) #for marmousi 101x101
        self.decoder_input = nn.Linear(latent_dim, filters[3]*13*25) #for marmousi 101x101
        #self.decoder_inputRho = nn.Linear(latent_dim, 1*300*100)
        
        self.up41 = autoUp5(filters[3], filters[3], self.is_deconv)
        self.up42 = autoUp5(filters[3], filters[3], self.is_deconv)
        
        #self.up4     = autoUp(filters[4], filters[3], self.is_deconv)
        self.up31     = autoUp5(filters[3], filters[2], self.is_deconv)
        #self.drop31   = nn.Dropout2d(0.1)
        self.up32     = autoUp5(int(filters[3]), int(filters[2]), self.is_deconv)
        #self.drop32   = nn.Dropout2d(0.1)
        #self.Rhoup33  = autoUp5(filters[3], int(filters[2]/4), self.is_deconv)
        #self.drop33   = nn.Dropout2d(0.1)
        #self.up3     = autoUp5(filters[3], filters[2], self.is_deconv)
        #self.dropU3  = nn.Dropout2d(0.025)
        self.up21     = autoUp5(filters[2], filters[1], self.is_deconv)
        #self.drop21   = nn.Dropout2d(0.1)
        self.up22     = autoUp5(int(filters[2]), int(filters[1]), self.is_deconv)
        #self.drop22   = nn.Dropout2d(0.1)
        #self.Rhoup23  = autoUp5(int(filters[2]/4), int(filters[1]/4), self.is_deconv)
        #self.drop23   = nn.Dropout2d(0.1)
        #self.up2     = autoUp5(filters[2], filters[1], self.is_deconv)
        #self.dropU2  = nn.Dropout2d(0.025)
        self.up11     = autoUp5(filters[1], filters[0], self.is_deconv)
        #self.drop11   = nn.Dropout2d(0.1)
        self.up12     = autoUp5(int(filters[1]), int(filters[0]), self.is_deconv)
        #self.drop12   = nn.Dropout2d(0.1)
        #self.Rhoup13  = autoUp5(int(filters[1]/4), int(filters[0]/4), self.is_deconv)
        #self.drop13   = nn.Dropout2d(0.1)
        #self.up1     = autoUp5(filters[1], filters[0], self.is_deconv)
        #self.dropU1  = nn.Dropout2d(0.025)
        ###self.upff1     = autoUp(filters[0], filters[0], self.is_deconv)
        ##self.upff2     = autoUp(filters[0], filters[0], self.is_deconv)
        #######self.f1      =  nn.Conv2d(filters[0],self.n_classes, 1)
        self.f11      =  nn.Conv2d(filters[0],int(filters[0]/2), 1)
        self.f12      =  nn.Conv2d(int(filters[0]),int(filters[0]/2), 1)
        #self.Rhof13      =  nn.Conv2d(int(filters[0]/4), int(filters[0]/8), 1)
        
        self.vp     =   nn.Conv2d(int(filters[0]/2),1,1)
        self.vs     =   nn.Conv2d(int(filters[0]/2),1,1)
        #self.Rhorho    =   nn.Conv2d(int(filters[0]/8), 1, 1)
        
        #self.final1   = nn.LeakyReLU(0.1)
        #self.final2   = nn.LeakyReLU(0.1)
        self.final1     =   nn.Sigmoid()
        self.final2     =   nn.Sigmoid()
        ##########self.final3     =   nn.Tanh()
        #self.f2      =  nn.Conv2d(1,1,1)
        #self.final1   =  nn.Sigmoid()
        #self.final1  =  nn.Conv2d(1, 1, 1)
        
    def forward(self, inputs1, inputs2, lstart, epoch1, latentI, lowf, inputs3, freq, idx, it):
        #filters = [16, 32, 64, 128, 256]
        #filters = [2, 4, 8, 16, 32]
        #filters = [32, 64, 128, 256, 512]
        #filters = [4,8,16,32,64]
        filters = [8, 16, 32, 64, 128]  ###this works very well
        #filters = [1, 1, 2, 4, 16]
        #filters = [16, 32, 64, 128, 256]
        #filters = [4, 8, 16, 32, 64]
        #filters = [2, 4, 8, 16, 32]
        latent_dim = 8
        label_dsp_dim = (190,348)
        #label_dsp_dim = (40,90)
        minvp = torch.min(inputs1[:,0,:,:])
        maxvp = torch.max(inputs1[:,0,:,:])
        
        minvs = torch.min(inputs1[:,1,:,:])
        maxvs = torch.max(inputs1[:,1,:,:])
        
        minrho = torch.min(inputs1[:,2,:,:])
        maxrho = torch.max(inputs1[:,2,:,:])
        
        #print("minrho :", minrho)
        #print("maxrho :", maxrho)
        
        #meandata = torch.mean(inputs2)
        #stddata = torch.std(inputs2)
        ############################################################
        combine1 = self.combine1((inputs2[:,:,1:5000:5,:]))
        combine2 = self.combine2((inputs3[:,:,1:5000:5,:]))
        
        c1c2 = torch.cat((combine1,combine2),axis=1)
        
        print("shape of inputs2 :", np.shape(inputs2))
        print("shape of inputs1 :", np.shape(inputs1))
        #down1  = self.down1((inputs2[:,:,1:1200:4,:]))
        down1  = self.down1(c1c2)
        #down1  = self.dropD1(down1)
        down2  = self.down2(down1)
        #down2  = self.dropD2(down2)
        down3  = self.down3(down2)
        #down3  = self.dropD3(down3)
        down4  = self.down4(down3)
        #down4  = self.dropD4(down4)
        
        print("shape of down4 :", np.shape(down4))
        
        #print("shape of down4 :", np.shape(down4))
        result = torch.flatten(down4, start_dim=1)
        
        #####print("result shape :", np.shape(result))
        
        p = self.decoder_input1(result)
        ###################################################################
        #p = inputs2
        #down3  = self.down3(down2)
        #down4  = self.down4(down3)s
        #center = self.center(down4)
        #up4    = self.up4(center)
        #up3    = self.up3(up4)
        #up2    = self.up2(up3)
        #print("shape of down 4:", np.shape(down2))
        #print("shape of result:", np.shape(result))
        #print("shape of p :", np.shape(p))
        latent1 = p
        
        #if (epoch1 <= lstart):
        #    latent1 = p
        #else:
        #    latent1 = latentI
        #    p = latent1
        #latent1 = p
            
        ########latent1 = p
        #p = inputs2
        #z = 0.5*torch.ones([1,1,1,64])
        z = self.decoder_input(p)
        ####zrho = self.decoder_inputRho(p)
        #####z = inputs2
        #z = z.view(-1, filters[3], 250, 51) #for marmousi model
        #print("shape of z :", np.shape(z))
        z = z.view(-1, filters[3], 13, 25)
        #zrho = zrho.view(-1, 1, 100, 300)
        
        up41    = self.up41(z)
        up42    = self.up42(z)
    
        up31    = self.up31(up41)
        #up31    = self.drop31(up31)
        up32    = self.up32(up42)
        #up32    = self.drop32(up32)
        ####up33    = self.Rhoup33(z)
        #up33    = self.drop33(up33)
        #up3      = self.up3(z)
        
        #up3    = self.dropU3(up3)
        #print(" shape of up1 :", np.shape(up1))
        up21    = self.up21(up31)
        #up21    = self.drop21(up21)
        up22    = self.up22(up32)
        #up22    = self.drop22(up22)
        ####up23    = self.Rhoup23(up33)
        #up23    = self.drop23(up23)
        #up2     = self.up2(up3)
        
        #up2    = self.dropU2(up2)
        up11    = self.up11(up21)
        #up11    = self.drop11(up11)
        up12    = self.up12(up22)
        #up12    = self.drop12(up12)
        ####up13    = self.Rhoup13(up23)
        #up13    = self.drop13(up13)
        #up1     = self.up1(up2)
        
        
        #up1    = self.dropU1(up1)
        print("shape of up11 :", np.shape(up11))
        print("shape of up12 :", np.shape(up12))
        up11    = up11[:,:,10:10+label_dsp_dim[0],10:10+label_dsp_dim[1]].contiguous()
        up12    = up12[:,:,10:10+label_dsp_dim[0],10:10+label_dsp_dim[1]].contiguous()
        ####up13    = up13[:,:,3:3+label_dsp_dim[0],3:3+label_dsp_dim[1]].contiguous()
        #up1    = up1[:,:,3:3+label_dsp_dim[0],3:3+label_dsp_dim[1]].contiguous()
        
        f11     = self.f11(up11)
        f12     = self.f12(up12)
        ####f13     = self.Rhof13(up13)
        #f1    = self.f1(up1)
        
        
        
        vp1f     = self.vp(f11)
        vs1f     = self.vs(f12)
        #####rho1f    = self.Rhorho(f13)
        #rho1    = self.rho2(rho1)
        ###vp1    = self.vp(torch.unsqueeze(f1[:,0,:,:],1))
        ###vs1    = self.vs(torch.unsqueeze(f1[:,1,:,:],1))
        #rho1   = self.rho(f13)
        #vp1     = f11
        #vs1     = f12
        #rho1    = f13
        
        #vp1f    = self.final1(vp1f)
        #vs1f    = self.final2(vs1f)
        ############rho1   = self.final3(rho1)
        #print("shape of vp1 :", np.shape(vp1))
        #vp1[:,:,0:15,:] = 0
        #vs1[:,:,0:15,:] = 0
        #rho1[:,:,0:15,:] = 0
        #rho1 = self.final3(rho1)
        #vp1f     = self.final1(vp1f)
        #vs1f     = self.final2(vs1f)
        
        print("maxvp :", maxvp)
        print("minvp :", minvp)
        print("maxvs :", maxvs)
        print("minvs :", minvs)
        #vp1    = 15.0 + vp1f*(maxvp-15.0)
        #vs1    = 1.0 + vs1f*(maxvs-1.0)
        #vp1 =  minvp + vp1f*(maxvp - minvp)
        #vs1 = 88.10 + vs1f*(maxvs - 88.10)
        vp1    = torch.unsqueeze(lowf[:,0,:,:],1) + vp1f
        vs1    = torch.unsqueeze(lowf[:,1,:,:],1) + vs1f
        ###############################rho1   = torch.unsqueeze(lowf[:,2,:,:],1)
        rho1   = torch.unsqueeze(lowf[:,2,:,:],1)

        #rho1    = self.final3(rho1)
        #vp1    = minvp + vp1*(maxvp-minvp)
        #vs1    = minvs + vs1*(maxvs-881.0)
        #rho1   = minrho + rho1*(maxrho-minrho)
        #vp1  = minvp + vp1f*(maxvp-minvp)
        #vs1  = minvs + vs1f*(maxvs-minvs)
        
        vp1    = torch.clip(vp1, min=minvp, max=maxvp)
        vs1    = torch.clip(vs1, min=49.81, max=maxvs)
        ####vp1 = minvp + vp1f*(maxvp-minvp)
        ####vs1  = 1.330 + vs1f*(maxvs-1.330)
        ####rho1   = torch.clip(rho1, min=17.199993, max=maxrho)
        #######vp1 = minvp + vp1*(maxvp-minvp)
        ########vs1 = minvs + vs1*(maxvs-minvs)
        ##########vs1 = 8.810*torch.ones((vs10.size())).cuda(vs10.get_device())
        
        vssmall = inputs1[:,1,:,:].cpu().numpy()
        vssmall = np.squeeze(vssmall)
        wb = 0*vssmall
        wb[(vssmall==0.0)]=1
        # #wb = np.flipud(wb)
        wb1 = np.ones(np.shape(wb))
        wb1 = 1-wb
        # #plt.imshow(wb1)
        nnz = np.zeros(348)
        # #print("shape of vp1 :", np.shape(vp1))
        for i in range(348):
            nnz[i] = int(np.max(np.nonzero(wb[:,i])))
        #     #print("nnz :", nnz[i])
        #     vp1[:,:,0:int(nnz[i]),i] = inputs1[:,0,0:int(nnz[i]),i]
        #     vs1[:,:,0:int(nnz[i]),i] = inputs1[:,1,0:int(nnz[i]),i]
        #nnz  = int(nnz)
        
        #vp1[:,:,0:24,:] = inputs1[:,0,0:24,:]
        #vs1[:,:,0:24,:] = inputs1[:,1,0:24,:]
        vswater = torch.unsqueeze(inputs1[:,1,:,:],0)
        print("shhape of vp1 :", np.shape(vp1))
        
        vp1[vswater==0] = 150.0
        vs1[vswater==0] = 0.0
        ################vp1[:,:,0:170,:] = inputs1[:,0,0:170,:]
        #####################vs1[:,:,0:170,:] = inputs1[:,1,0:170,:]
        ####rho1[:,:,0:25,:] = inputs1[:,2,0:25,:]
        
        
       #vp1     = inputs1[:,0,:,:]
        #rho1     = inputs1[:,2,:,:]
        
        
        #vp1    = torch.unsqueeze(vp1,1)
        #vs1    = torch.unsqueeze(vs1,1)
        #rho1   = torch.unsqueeze(rho1,1)
        #f11    = torch.cat((vp1,vs1),dim=1)
        #f11     = vp1
        #f1     = self.final(f1)
        #f1     = self.final1(f1)
        #f1     = self.final(f1)
        #f1     = f1/torch.max(f1)
        #print("mintrue :", mintrue)
        #print("maxtrue :", maxtrue)
        
        #f1    = mintrue + f1*(maxtrue-mintrue)
        #f1[(inputs1==1500)] = 1500
        #f1     = lowf + f1
        #f1[(inputs1 == 1.510)] = 1.510
        #f1     = torch.clamp(f1,min=mintrue,max=maxtrue)
        #f1[(inputs1 == 1.510)] = 1.510
        
        #f1     = torch.add(f1,1600.0)
        #f1     = torch.add(f1,lowf)
        #f1     = 3.0 + f1*(6.0-3.0)
        #f1     = torch.clamp(f1, min=mintrue, max=maxtrue)
        #print("shape of f1 :", np.shape(f1))
        #f1[(inputs1==2000)] = 2000
        #f1     = f1*100
        #f1     = torch.clip(f1, min=1500, max=3550) ##clamping for marmousi
        #with torch.no_grad():
        #    f4 = torch.clamp(f1,15.0, 35.5)  # You must use v[:]=xxx instead of v=xxx
        #f1[:,:,0:26,:] = 1500.0
        #f1     = torch.clamp(f1,min=20,max=45)
        latent1 = 0
        grad = 0*vp1
        lossT = 0.0
        vp_grad = vp1*0
        vs_grad = vp1*0
        rho_grad = vp1*0
        
        #vs1 = vp1*0
        #rho1 = vp1*0
        if (epoch1 > lstart):
            [vp_grad, vs_grad, rho_grad, lossT] = self.prop(vp1, vs1, rho1, inputs1, epoch1, freq, idx, it, nnz)
        #if (epoch1 > lstart):
        #    [grad, lossT] = self.prop(inputs2, f1, lstart, epoch1, mintrue, maxtrue, inputs1)
        #    grad = grad.to(inputs2.get_device())
        #    grad = torch.unsqueeze(grad,0)
        #    grad = torch.unsqueeze(grad,0)
        #result = torch.flatten(f1, start_dim=1)
        #print(" shape of grad :", np.shape(grad))

        return vp1, vs1, rho1, grad, latent1, vp_grad, vs_grad, rho_grad, lossT
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                    
                    
class SIMPLENET(nn.Module):
    def __init__(self,outer_nc=30, inner_nc=1, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(SIMPLENET, self).__init__()
        self.is_deconv     = False
        self.in_channels   = outer_nc
        self.is_batchnorm  = True
        self.n_classes     = inner_nc
        
        #filters = [16, 32, 64, 128, 512]
        #filters = [2, 4, 8, 16, 32]
        filters = [8, 16, 32, 64, 128]
        
        latent_dim = 8

        self.down1   = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        #self.drop1   = nn.Dropout2d(0.1)
        self.down2   = unetDown(filters[0], filters[1], self.is_batchnorm)
        #self.drop2   = nn.Dropout2d(0.1)
        self.down3   = unetDown(filters[1], filters[2], self.is_batchnorm)
        #self.drop3   = nn.Dropout2d(0.1)
        self.down4   = unetDown(filters[2], filters[3], self.is_batchnorm)
        # self.center  = unetConv2(filters[3], filters[4], self.is_batchnorm)
        ##self.decoder_input1 = nn.Linear(filters[1]*250*51, latent_dim) #for marmousi 151x200
        #self.decoder_input1 = nn.Linear(filters[2]*125*26, latent_dim) #for marmousi 151x200
        #self.decoder_input = nn.Linear(latent_dim, filters[2]*500*102) #for marmousi 151x200
        self.decoder_input1 = nn.Linear(filters[3]*63*13, latent_dim) #for marmousi 101x101
        #self.decoder_input = nn.Linear(latent_dim, filters[3]*100*26) #for marmousi 101x101
        #self.decoder_input1 = nn.Linear(filters[1]*100*18, latent_dim) #for marmousi 101x101
        self.decoder_input = nn.Linear(latent_dim, filters[3]*25*19) #for marmousi 101x101
        
        
        #self.up4     = autoUp(filters[4], filters[3], self.is_deconv)
        self.up3     = autoUp5(filters[3], filters[2], self.is_deconv)
        self.up2     = autoUp5(filters[2], filters[1], self.is_deconv)
        self.up1     = autoUp5(filters[1], filters[0], self.is_deconv)
        #self.upff1     = autoUp(filters[0], filters[0], self.is_deconv)
        #self.upff2     = autoUp(filters[0], filters[0], self.is_deconv)
        self.f1      =  nn.Conv2d(filters[0],self.n_classes, 1)
        #self.f2      =  nn.Conv2d(1,1,1)
        self.final   =  nn.Sigmoid()
        #self.final1  =  nn.Conv2d(1, 1, 1)
        
    def forward(self, inputs1, inputs2, lstart, epoch1, latentI, lowf):
        #filters = [16, 32, 64, 128, 512]
        filters = [8, 16, 32, 64, 128]
        latent_dim = 8
        label_dsp_dim = (151,200)
        mintrue = torch.min(inputs1)
        maxtrue = torch.max(inputs1)
        meandata = torch.mean(inputs2)
        stddata = torch.std(inputs2)
        down1  = self.down1((inputs2[:,:,1:4001:4,:]))
        #down1  = self.drop1(down1)
        down2  = self.down2(down1)
        #down2  = self.drop2(down2)
        down3  = self.down3(down2)
        #down3  = self.drop3(down3)
        down4  = self.down4(down3)
        
        #print("shape of down3 :", np.shape(down))
        
        #print("shape of down2 :", np.shape(down2))
        result = torch.flatten(down4, start_dim=1)
        
        #print("result shape :", np.shape(result))
        
        p = self.decoder_input1(result)
        #down3  = self.down3(down2)
        #down4  = self.down4(down3)s
        #center = self.center(down4)
        #up4    = self.up4(center)
        #up3    = self.up3(up4)
        #up2    = self.up2(up3)
        #print("shape of down 4:", np.shape(down2))
        #print("shape of result:", np.shape(result))
        latent1 = p
        #if (epoch1 <= lstart):
        #    latent1 = p
        #else:
        #    latent1 = latentI
        #    p = latent1
        #    latent1 = p
            

        #p = torch.randn([1,1,1,8])
        #z = 0.5*torch.ones([1,1,1,64])
        z = self.decoder_input(p)
        #z = z.view(-1, filters[3], 250, 51) #for marmousi model
        z = z.view(-1, filters[3], 19, 25)
    
        up3    = self.up3(z)
        #print(" shape of up1 :", np.shape(up1))
        up2    = self.up2(up3)
        up1    = self.up1(up2)
        print("shape of up1 :", np.shape(up1))
        up1    = up1[:,:,1:1+label_dsp_dim[0],0:1+label_dsp_dim[1]].contiguous()
        f1     = self.f1(up1)
        f1     = self.final(f1)
        #f1     = self.final1(f1)
        #f1     = self.final(f1)
        #f1     = f1/torch.max(f1)
        #print("shape of f1 :", np.shape(f1))
        print("mintrue :", mintrue)
        print("maxtrue :", maxtrue)
        
        f1    = mintrue + f1*(maxtrue-mintrue)
        f1[(inputs1==1500)] = 1500
        #f1     = lowf + f1
        #f1[(inputs1 == 1.510)] = 1.510
        #f1     = torch.clamp(f1,min=mintrue,max=maxtrue)
        #f1[(inputs1 == 1.510)] = 1.510
        
        #f1     = torch.add(f1,1600.0)
        #f1     = torch.add(f1,lowf)
        #f1     = 3.0 + f1*(6.0-3.0)
        #f1     = torch.clamp(f1, min=mintrue, max=maxtrue)
        #print("shape of f1 :", np.shape(f1))
        #f1[(inputs1==2000)] = 2000
        #f1     = f1*100
        #f1     = torch.clip(f1, min=1500, max=3550) ##clamping for marmousi
        #with torch.no_grad():
        #    f4 = torch.clamp(f1,15.0, 35.5)  # You must use v[:]=xxx instead of v=xxx
        #f1[:,:,0:26,:] = 1500.0
        #f1     = torch.clamp(f1,min=20,max=45)
        
        grad = 0*f1
        lossT = 0.0
        if (epoch1 > lstart):
            [lossT] = self.prop(inputs2, f1, lstart, epoch1, mintrue, maxtrue, inputs1)
            #grad = grad.to(inputs2.get_device())
            #grad = torch.unsqueeze(grad,0)
            #grad = torch.unsqueeze(grad,0)
        #result = torch.flatten(f1, start_dim=1)
        #print(" shape of grad :", np.shape(grad))

        return f1, grad, latent1, lossT, down3, up2, up1
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    # forward modeling to compute gradients
    def prop(self, inputs, vel, lstart, epoch1, mintrue, maxtrue, true):
        
        torch.cuda.set_device(5)  #RB Necessary if device <> 0
        GPU_string='cuda:'+str(5)
        devicek = torch.device(GPU_string)
        #vel = vel.to(devicek)
        #net1out1 = mintrue + vel*(maxtrue-mintrue)
        net1out1 = vel
        #net1out1 = net1out2.to(devicek)
        #net1out1 = (3550-1500)*vel+1500
        #print("---shape of vel---", str(np.shape(vel)))
        #print("devicek :", devicek)
        net1out1 = net1out1.detach()
        net1out1 = torch.squeeze(net1out1)
        g1 = torch.arange(net1out1.size(dim=0))
        g1 = g1**2.0
        ss = g1.tile((200,1))
        ss = torch.transpose(ss,0,1)
        net1out1 = net1out1.to(devicek)
        #devicek = net1out1.get_device()
        #net1out1[0:26,:] = 1500.0

        
        freq = 8
        dx = 10
        nt = 4001
        dt = 0.001
        num_shots = 30
        num_receivers_per_shot = 200
        num_sources_per_shot = 1
        num_dims = 2
        #ModelDim = [201,301]
        ny = 200
        source_spacing = 200 * dx / num_shots
        receiver_spacing = 200 * dx / num_receivers_per_shot
        x_s = torch.zeros(num_shots, num_sources_per_shot, num_dims)
        x_s[:, 0, 1] = torch.linspace(0,(ny-1)*dx,num_shots)
        x_r = torch.zeros(num_shots, num_receivers_per_shot, num_dims)
        x_r[0, :, 1] = torch.arange(num_receivers_per_shot).float() * receiver_spacing
        x_r[:, :, 1] = x_r[0, :, 1].repeat(num_shots, 1)

        source_amplitudes_true = (deepwave.wavelets.ricker(freq, nt, dt, 1/freq)
                                  .reshape(-1, 1, 1))
        #print("device ordinal :", self.devicek)
        source_amplitudes_true = source_amplitudes_true.to(devicek)
        #lstart = -1
        num_batches = 3
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
        ss = ss.to(devicek)
        ################data misfit calculation##########################################

        #net1out1 = net1out1.to(self.devicek)

        receiver_amplitudes_true = inputs[0,:,:,:]
        receiver_amplitudes_true = receiver_amplitudes_true.swapaxes(0,1)
        receiver_amplitudes_true = receiver_amplitudes_true.to(devicek)
        #print("shape of receiver amplitudes true")
        #print(np.shape(receiver_amplitudes_true))

        ######rcv_amps_true_norm = receiver_amplitudes_true
        #print("receiver amplitude true shape")
        # print(np.shape(receiver_amplitudes_true))
        #net1out1 = net1out.detach()
        #net1out1 = torch.tensor(net1out1)
        #net1out1 = net1out1*(4500-2000)+2000
        #print(np.shape(net1out1))
        #min1 = torch.min(net1out1)
        #print("min1 :", min1)
        #print(min1.get_device())
        #min1 = min1.to(self.device1)
        mat2 = torch.ones(net1out1.size()[0],net1out1.size()[1]).to(devicek)
        mat2 = mat2 * 1500
        #mat2 = torch.clamp(mat2,min=1500,max=3550)
        #min1 = torch.min(net1out1)
        #max1 = torch.max(net1out1)
        #if (epoch1 == 52): 
        #np.save('./deepwave/before1.npy',net1out1.cpu().detach().numpy())
        # np.save('ftout1',net1out1.cpu().numpy())
        #net1out1 = net1out1.to(self.devicek)
        #mat2 = mat2.to(self.devicek)
        src_amps = source_amplitudes_true.repeat(
                        1, num_shots, 1)
        prop2 = deepwave.scalar.Propagator({'vp': mat2}, dx)
        receiver_amplitudes_cte = prop2(src_amps,
                                x_s.to(devicek),
                                x_r.to(devicek), dt)
        
        receiver_amplitudes_true = receiver_amplitudes_true - receiver_amplitudes_cte
        
        #print("receiver_amplitudes_true :", np.shape(receiver_amplitudes_true))
        #print("receiver_amplitudes_cte :", np.shape(receiver_amplitudes_cte))
        #receiver_amplitudes_true = receiver_amplitudes_true
        rcv_amps_true_max, _ = torch.abs(receiver_amplitudes_true).max(dim=0, keepdim=True)
        #rcv_amps_true_norm = receiver_amplitudes_true / (rcv_amps_true_max.abs() + 1e-10)
        rcv_amps_true_norm = receiver_amplitudes_true

        criterion1 = torch.nn.L1Loss()
        #vgg = Vgg16().type(torch.cuda.FloatTensor)
        #criterion2 = torch.nn.MSELoss()
        #print("shape of mat2 :", np.shape(mat2))
        

        if (epoch1 > lstart):
            net1out1.requires_grad = True
            optimizer2 = torch.optim.Adam([{'params': [net1out1], 'lr':10}])
            
        model2 = net1out1.clone()
        model2 = torch.clamp(net1out1,min=mintrue,max=maxtrue)
        
        prop = deepwave.scalar.Propagator({'vp':model2}, dx)
        receiver_amplitudes_pred = prop(source_amplitudes_true,
                                x_s,
                                x_r, dt)
        receiver_amplitudes_pred = receiver_amplitudes_pred - receiver_amplitudes_cte
        
        lossinner = criterion1(receiver_amplitudes_true, receiver_amplitudes_pred)

        # for epoch in range(num_epochs):
        #         #Shuffle shot coordinates
        #         idx = torch.randperm(num_shots)
        #         #idx = idx.type(torch.LongTensor)
        #         x_s = x_s.view(-1,2)[idx].view(x_s.size())
        #         #RB Shuffle true's seismograms sources with same random values
        #         rcv_amps_true_norm = rcv_amps_true_norm[:,idx,:]
        #         #RB Shuffle direct wave seismograms sources with the same random values
        #         receiver_amplitudes_cte = receiver_amplitudes_cte[:,idx,:]
        
        #         for it in range(num_batches):
        #             #if (epoch1 > lstart):
        #             optimizer2.zero_grad()
        #             model2 = net1out1.clone()
        #             model2 = torch.clamp(net1out1,min=mintrue,max=maxtrue)
        #             #np.save('before108.npy',net1out1.cpu().detach().numpy())
        #             #net1out1 = torch.clamp(net1out1,min=2000,max=4500)
        #             prop = deepwave.scalar.Propagator({'vp': model2}, dx)
        #             batch_src_amps = source_amplitudes_true.repeat(1, num_shots_per_batch, 1)
        #             #print("shape of batch src amps")
        #             #print(np.shape(batch_src_amps))
        #             ############batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches].to(self.devicek)
        #             batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches]
        #             batch_rcv_amps_cte = receiver_amplitudes_cte[:,it::num_batches]
        #             batch_x_s = x_s[it::num_batches].to(devicek)
        #             ##################batch_x_s = x_s[it::num_batches]
        #             #print("shape of batch src amps")
        #             # print(np.shape(batch_x_s))
        #             #####################batch_x_r = x_r[it::num_batches]
        #             batch_x_r = x_r[it::num_batches].to(devicek)
        #             #print("shape of batch receiver amps")
        #             # print(np.shape(batch_x_r))
        #             batch_rcv_amps_pred = prop(batch_src_amps, batch_x_s, batch_x_r, dt)
        #             #print("batch_rcv_amps_pred")
        #             #print(np.shape(batch_rcv_amps_pred))
        #             batch_rcv_amps_pred = batch_rcv_amps_pred - batch_rcv_amps_cte
        #             batch_rcv_amps_pred_max, _ = torch.abs(batch_rcv_amps_pred).max(dim=0, keepdim=True)
        #             # Normalize amplitudes by dividing by the maximum amplitude of each receiver
        #             #batch_rcv_amps_pred_norm = batch_rcv_amps_pred / (batch_rcv_amps_pred_max.abs() + 1e-10)
        #             batch_rcv_amps_pred_norm = batch_rcv_amps_pred 
        #             ##############batch_rcv_amps_pred_norm = batch_rcv_amps_pred
                    
        #             #print("shape of receiver amplitudes predicted")
        #             # print(np.shape(batch_rcv_amps_pred))
        #             lossinner1 = criterion1(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
        #             #lossinner2 = criterion2(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
        #             lossinner = lossinner1
        #             #y_c_features = vgg(torch.unsqueeze(batch_rcv_amps_true,0))
        #             #########model2.grad[0:26,:] = 0
        #             #filen = './deepwave/epoch1'+str(epoch)+'.npy'
        #             #np.save(filen,net1out1.cpu().detach().numpy())
        #             ##############if (epoch == num_epochs-1):
        #             ##########    sumlossinner += lossinner.item()
        #             #########if (epoch1 > lstart):
        #             lossinner.backward()
        #             net1out1.grad = net1out1.grad * ss
        #             net1out1.grad[(true[0,0,:,:]==1500)] = 0
                    #net1out1.grad[0:26,:] = 0
                    ##########optimizer2.step()
                    #epoch_loss += loss.item()
                    #optimizer2.step()
        #if (epoch1 == 52): 
        #print("shape of inputs :", np.shape(inputs))
        #np.save('./marmousi/rcv_amplitudes.npy',batch_rcv_amps_pred.cpu().detach().numpy())
        #np.save('./marmousi/rcv_amplitudes_true.npy',batch_rcv_amps_true.cpu().detach().numpy())
        #np.save('./marmousi/rcv_amplitudes_true_cte.npy',batch_rcv_amps_cte.cpu().detach().numpy())
        #np.save('./marmousi/net1o420ut1.npy',net1out1.cpu().detach().numpy())
        #np.save('./marmousi/netgrad1.npy',net1out1.grad.cpu().detach().numpy())
        #np.save('./deepwave/seis231.npy',batch_rcv_amps_pred.cpu().detach().numpy())
        #net1out1 = (net1out1 - 2000)/(4500-2000)
        #net1out1 = (net1out1-2000)/(4500-2000)
        #net1out1.grad = net1out1.grad*1000
                 
        return lossinner.item()