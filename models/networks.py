import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import sys
import torch.nn.functional as F
import deepwave
#import torchgeometry as tgm
from torchvision import models

sys.path.append('./models')

from resunet_modules import (
    ResidualConv,
    ASPP,
    AttentionBlock,
    Upsample_,
    Squeeze_Excite_Block,
)
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
#from unet_layers import unetConv2
import torch.distributions.transforms as transform
import torch.distributions as distrib

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        print(gpu_ids)
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'classic':
        net = ClassicU_Net(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'Att':
        net = AttU_Net(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)  
    elif netG == 'NewU':
        net = NewU_Net(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout) 
    elif netG == 'Auto':
        net = AutoMarmousi_Net(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'Auto21':
        net = AutoMarmousi21_Net(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'Auto22':
        net = AutoMarmousi22_Net(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'Auto23':
        net = AutoMarmousi23_Net(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'Vae':
        net = Vae_Net(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout) 
    elif netG == 'Vae2':
        net = VaeMarmousi_Net(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout) 
    elif netG == 'Vae3':
        net = VaeMarmousi3_Net(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'VaeNoPhy':
        net = VaeNoPhy_Net(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout) 
    elif netG == 'VaeLatentNoPhy':
        net = VaeLatentNoPhy_Net(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout) 
    elif netG == 'VaeLatent2NoPhy':
        net = VaeLatent2NoPhy_Net(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout) 
    elif netG == 'VaeNormalizing':
        net = VaeNormalizing_Net(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout) 
    elif netG == 'VaeNormalizingPhy':
        net = VaeNormalizingPhy_Net(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout) 
    elif netG == 'Vaevel':
        net = Vaevel_Net(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout) 
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            #print("shape of target tensor")
            #print(target_tensor)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.LeakyReLU(0.5)]
        lastll1 = nn.Conv2d(output_nc, output_nc, kernel_size=3,
                              stride=1, padding=1)
        model += [lastll1]
        lastll2 = nn.Conv2d(output_nc, output_nc, kernel_size=3,
                              stride=1, padding=1)
        model += [lastll2]
        #model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class UnetAttGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetAttGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetAttSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetAttSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetAttSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetAttSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetAttSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetAttSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.attb = Attention_block(F_g=outer_nc, F_l=outer_nc, F_int=int(outer_nc/2))
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            lastll1 = nn.Conv2d(outer_nc, outer_nc, kernel_size=3,
                              stride=1, padding=1)
            lastll2 = nn.Conv2d(outer_nc, outer_nc, kernel_size=3,
                              stride=1, padding=1)
            down = [downconv]
            #up = [uprelu, upconv, nn.Tanh(),lastll1,lastll2]
            up = [uprelu, upconv, nn.LeakyReLU(0.8),lastll1,lastll2]
            #up = [uprelu, upconv]

            model = down + [submodule] + up + self.attb
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            print("--model---")
            print(down)
            print(type(down))
            print("---attb---")
            print(self.attb)
            print(type(self.attb))
            model = down + up + self.attb
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)

            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + self.attb + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up + self.attb

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            y = self.model(x)
            x = self.attb(g=y,x=x)
            return torch.cat([x, y], 1)



class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            lastll1 = nn.Conv2d(outer_nc, outer_nc, kernel_size=3,
                              stride=1, padding=1)
            lastll2 = nn.Conv2d(outer_nc, outer_nc, kernel_size=3,
                              stride=1, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            #up = [uprelu, upconv, nn.LeakyReLU(0.8),lastll1,lastll2]
            #up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2,inplace=True),
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class conv_block_sqex(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=4,stride=2,padding=1,bias=False),
            Squeeze_Excite_Block(ch_out),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2,inplace=True),
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch_in,ch_out,kernel_size=4,stride=2,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
            #nn.Dropout(0.5, inplace=True),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class ClassicU_Net(nn.Module):
    def __init__(self,outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ClassicU_Net,self).__init__()
        
        #self.Con = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Conv1 = nn.Conv2d(outer_nc,32,kernel_size=4,stride=2,padding=1,bias=False)
        self.ReLU1 = nn.LeakyReLU(0.2,True)

        self.Conv2 = conv_block(ch_in=32,ch_out=64)

        self.Conv3 = conv_block(ch_in=64,ch_out=128)

        self.Conv4 = conv_block(ch_in=128,ch_out=256)

        self.Conv5 = conv_block(ch_in=256,ch_out=256)

        self.Conv6 = nn.Conv2d(256,256,kernel_size=4,stride=2,padding=1,bias=False)
        self.ReLU2 = nn.ReLU(inplace=True)

        self.Up5 = up_conv(ch_in=256,ch_out=256)

        self.Up4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.Bn1 = nn.BatchNorm2d(256)
        self.Dp1 = nn.Dropout(0.5,inplace=False)
        self.ReLU4 = nn.ReLU(inplace=True)

        self.Up3 = up_conv(ch_in=512,ch_out=128)

        self.Up2 = up_conv(ch_in=256,ch_out=64)

        self.Up1 = up_conv(ch_in=128,ch_out=32)

        self.F1 = nn.ConvTranspose2d(64,1,kernel_size=4,stride=2,padding=1,bias=True)
        self.ReLU3 = nn.LeakyReLU(0.8,True)

        self.F2 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        self.F3 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)



    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x1 = self.ReLU1(x1)
        x2 = self.Conv2(x1)
        x3 = self.Conv3(x2)
        x4 = self.Conv4(x3)
        x5 = self.Conv5(x4)
        x6 = self.Conv6(x5)
        x6 = self.ReLU2(x6)

        # decoding + concat path
        d5 = self.Up5(x6)
        d5 = torch.cat((x5,d5),dim=1)
        
        d4 = self.Up4(d5)
        d4 = self.Bn1(d4)
        d4 = self.Dp1(d4)
        d4 = self.ReLU4(d4)
        d4 = torch.cat((x4,d4),dim=1)

        d3 = self.Up3(d4)
        d3 = torch.cat((x3,d3),dim=1)

        d2 = self.Up2(d3)
        d2 = torch.cat((x2,d2),dim=1)

        d1 = self.Up1(d2)
        d1 = torch.cat((x1,d1),dim=1)
        
        f1 = self.F1(d1)
        f1 = self.ReLU3(f1)

        f2 = self.F2(f1)
        f3 = self.F3(f2)


        return f3



class AttU_Net(nn.Module):
    def __init__(self,outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(AttU_Net,self).__init__()
        
        #self.Con = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Conv1 = nn.Conv2d(outer_nc,32,kernel_size=4,stride=2,padding=1,bias=False)
        self.ReLU1 = nn.LeakyReLU(0.2,True)

        self.Conv2 = conv_block(ch_in=32,ch_out=64)

        self.Conv3 = conv_block(ch_in=64,ch_out=128)

        self.Conv4 = conv_block(ch_in=128,ch_out=256)

        self.Conv5 = conv_block(ch_in=256,ch_out=256)

        self.Conv6 = nn.Conv2d(256,256,kernel_size=4,stride=2,padding=1,bias=False)
        self.ReLU2 = nn.ReLU(inplace=True)

        self.Up5 = up_conv(ch_in=256,ch_out=256)
        self.Att5 = Attention_block(F_g=256,F_l=256,F_int=128)

        self.Up4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.Bn1 = nn.BatchNorm2d(256)
        self.Dp1 = nn.Dropout(0.5,inplace=False)
        self.ReLU4 = nn.ReLU(inplace=True)

        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)

        self.Up3 = up_conv(ch_in=512,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)

        self.Up2 = up_conv(ch_in=256,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)

        self.Up1 = up_conv(ch_in=128,ch_out=32)
        self.Att1 = Attention_block(F_g=32,F_l=32,F_int=16)

        self.F1 = nn.ConvTranspose2d(64,1,kernel_size=4,stride=2,padding=1,bias=True)
        self.ReLU3 = nn.LeakyReLU(0.8,True)

        self.F2 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        self.F3 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)



    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x1 = self.ReLU1(x1)
        x2 = self.Conv2(x1)
        x3 = self.Conv3(x2)
        x4 = self.Conv4(x3)
        x5 = self.Conv5(x4)
        x6 = self.Conv6(x5)
        x6 = self.ReLU2(x6)

        # decoding + concat path
        d5 = self.Up5(x6)
        x5 = self.Att5(g=d5,x=x5)
        d5 = torch.cat((x5,d5),dim=1)
        
        d4 = self.Up4(d5)
        d4 = self.Bn1(d4)
        d4 = self.Dp1(d4)
        d4 = self.ReLU4(d4)
        x4 = self.Att4(g=d4,x=x4)
        d4 = torch.cat((x4,d4),dim=1)

        d3 = self.Up3(d4)
        x3 = self.Att3(g=d3,x=x3)
        d3 = torch.cat((x3,d3),dim=1)

        d2 = self.Up2(d3)
        x2 = self.Att2(g=d2,x=x2)
        d2 = torch.cat((x2,d2),dim=1)

        d1 = self.Up1(d2)
        x1 = self.Att1(g=d1,x=x1)
        d1 = torch.cat((x1,d1),dim=1)
        
        f1 = self.F1(d1)
        f1 = self.ReLU3(f1)

        f2 = self.F2(f1)
        f3 = self.F3(f2)


        return f3


class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class conv_block_sqex(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block_sqex,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=4,stride=2,padding=1,bias=False),
            Squeeze_Excite_Block(ch_out),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2,inplace=True),
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class ASPPU_Net(nn.Module):
    def __init__(self,outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ASPPU_Net,self).__init__()
        
        #self.Con = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Conv1 = nn.Conv2d(outer_nc,32,kernel_size=4,stride=2,padding=1,bias=False)
        self.ReLU1 = nn.LeakyReLU(0.2,True)

        self.Conv2 = conv_block_sqex(ch_in=32,ch_out=64)

        self.Conv3 = conv_block_sqex(ch_in=64,ch_out=128)

        self.Conv4 = conv_block_sqex(ch_in=128,ch_out=256)

        self.Conv5 = conv_block_sqex(ch_in=256,ch_out=256)

        self.Conv6 = nn.Conv2d(256,256,kernel_size=4,stride=2,padding=1,bias=False)
        self.ReLU2 = nn.ReLU(inplace=True)

        self.ASPP = ASPP(256,256)
        #self.ReLU2 = nn.ReLU(inplace=True)

        self.Up5 = up_conv(ch_in=256,ch_out=256)

        self.Up4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.Bn1 = nn.BatchNorm2d(256)
        self.Dp1 = nn.Dropout(0.5,inplace=False)
        self.ReLU4 = nn.ReLU(inplace=True)

        self.Up3 = up_conv(ch_in=512,ch_out=128)

        self.Up2 = up_conv(ch_in=256,ch_out=64)

        self.Up1 = up_conv(ch_in=128,ch_out=32)

        self.F1 = nn.ConvTranspose2d(64,inner_nc,kernel_size=4,stride=2,padding=1,bias=True)
        self.ReLU3 = nn.LeakyReLU(0.8,True)

        #self.F2 = nn.Conv2d(inner_nc,inner_nc,kernel_size=3,stride=1,padding=1)
        #self.F3 = nn.Conv2d(inner_nc,inner_nc,kernel_size=3,stride=1,padding=1)



    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x1 = self.ReLU1(x1)
        x2 = self.Conv2(x1)
        x3 = self.Conv3(x2)
        x4 = self.Conv4(x3)
        x5 = self.Conv5(x4)
        x6 = self.Conv6(x5)
        x6 = self.ReLU2(x6)

        aspp = self.ASPP(x6)        
        # decoding + concat path
        d5 = self.Up5(aspp)
        d5 = torch.cat((x5,d5),dim=1)
        
        d4 = self.Up4(d5)
        d4 = self.Bn1(d4)
        d4 = self.Dp1(d4)
        d4 = self.ReLU4(d4)
        d4 = torch.cat((x4,d4),dim=1)

        d3 = self.Up3(d4)
        d3 = torch.cat((x3,d3),dim=1)

        d2 = self.Up2(d3)
        d2 = torch.cat((x2,d2),dim=1)

        d1 = self.Up1(d2)
        d1 = torch.cat((x1,d1),dim=1)
        
        f1 = self.F1(d1)
        f1 = self.ReLU3(f1)

        #f2 = self.F2(f1)
        #f3 = self.F3(f2)


        return f1


#################ResUnetPlusPlus#######################################
class ResUnetPlusPlus_Net(nn.Module):
    def __init__(self,outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, filters=[64, 128, 256, 512, 1024]):
        super(ResUnetPlusPlus_Net, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(outer_nc, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(outer_nc, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])

        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])

        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], inner_nc, 1),nn.LeakyReLU(0.8,inplace=True))

        self.F2 = nn.Conv2d(inner_nc,inner_nc,kernel_size=3,stride=1,padding=1)
        self.F3 = nn.Conv2d(inner_nc,inner_nc,kernel_size=3,stride=1,padding=1)

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        x10 = self.output_layer(x9)

        x11 = self.F2(x10)
        out = self.F3(x11)

        return out

###########Mutli UNET : UNET with multiple outputs####################################
class MultiU_Net(nn.Module):
    def __init__(self,outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(MultiU_Net,self).__init__()
        
        #self.Con = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Conv1 = nn.Conv2d(outer_nc,32,kernel_size=4,stride=2,padding=1,bias=False)
        self.ReLU1 = nn.LeakyReLU(0.2,True)

        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv21 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.ReLU21 = nn.LeakyReLU(0.2,True)

        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv31 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        self.ReLU31 = nn.LeakyReLU(0.2,True)

        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv41 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.ReLU41 = nn.LeakyReLU(0.2,True)

        self.Conv5 = conv_block(ch_in=256,ch_out=256)
        self.Conv51 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.ReLU51 = nn.LeakyReLU(0.2,True)

        self.Conv6 = nn.Conv2d(256,256,kernel_size=4,stride=2,padding=1,bias=False)
        self.ReLU2 = nn.ReLU(inplace=True)
        self.Conv61 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.ReLU61 = nn.ReLU(inplace=True)

        self.Up5 = up_conv(ch_in=256,ch_out=256)
        self.Up51 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
        self.ReLU51up = nn.ReLU(inplace=True)

        self.Up4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.Bn1 = nn.BatchNorm2d(256)
        self.Dp1 = nn.Dropout(0.5,inplace=False)
        self.ReLU4 = nn.ReLU(inplace=True)
        self.Up41 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
        self.ReLU41up = nn.ReLU(inplace=True)

        self.Up3 = up_conv(ch_in=512,ch_out=128)
        self.Up31 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.ReLU31up = nn.ReLU(inplace=True)

        self.Up2 = up_conv(ch_in=256,ch_out=64)
        self.Up21 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        self.ReLU21up = nn.ReLU(inplace=True)

        self.Up11 = up_conv(ch_in=128,ch_out=32)
        self.Up12 = up_conv(ch_in=128,ch_out=32)
        self.Up13 = up_conv(ch_in=128,ch_out=32)

        self.F11 = nn.ConvTranspose2d(64,1,kernel_size=4,stride=2,padding=1,bias=True)
        self.ReLU31 = nn.LeakyReLU(0.8,True)
        self.F12 = nn.ConvTranspose2d(64,1,kernel_size=4,stride=2,padding=1,bias=True)
        self.ReLU32 = nn.LeakyReLU(0.8,True)
        self.F13 = nn.ConvTranspose2d(64,1,kernel_size=4,stride=2,padding=1,bias=True)
        self.ReLU33 = nn.LeakyReLU(0.8,True)

        self.F21 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        self.F22 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        self.F23 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)

        self.F31 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        self.F32 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        self.F33 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)



    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x1 = self.ReLU1(x1)

        x2 = self.Conv2(x1)
        x2 = self.Conv21(x2)
        x2 = self.ReLU21(x2)

        x3 = self.Conv3(x2)
        x3 = self.Conv31(x3)
        x3 = self.ReLU31(x3)

        x4 = self.Conv4(x3)
        x4 = self.Conv41(x4)
        x4 = self.ReLU41(x4)

        x5 = self.Conv5(x4)
        x5 = self.Conv51(x5)
        x5 = self.ReLU51(x5)


        x6 = self.Conv6(x5)
        x6 = self.ReLU2(x6)
        x6 = self.Conv61(x6)
        x6 = self.ReLU61(x6)

        # decoding + concat path
        d5 = self.Up5(x6)
        d5 = torch.cat((x5,d5),dim=1)
        d5 = self.Up51(d5)
        d5 = self.ReLU51up(d5)
        
        d4 = self.Up4(d5)
        d4 = self.Bn1(d4)
        d4 = self.Dp1(d4)
        d4 = self.ReLU4(d4)
        d4 = torch.cat((x4,d4),dim=1)
        d4 = self.Up41(d4)
        d4 = self.ReLU41up(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x3,d3),dim=1)
        d3 = self.Up31(d3)
        d3 = self.ReLU41up(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x2,d2),dim=1)
        d2 = self.Up21(d2)
        d2 = self.ReLU21up(d2)

        d11 = self.Up11(d2)
        d11 = torch.cat((x1,d11),dim=1)
        d12 = self.Up12(d2)
        d12 = torch.cat((x1,d12),dim=1)
        d13 = self.Up13(d2)
        d13 = torch.cat((x1,d13),dim=1)
        
        f11 = self.F11(d11)
        f11 = self.ReLU31(f11)
        f12 = self.F12(d12)
        f12 = self.ReLU32(f12)
        f13 = self.F13(d13)
        f13 = self.ReLU33(f13)

        f21 = self.F21(f11)
        f22 = self.F22(f12)
        f23 = self.F23(f13)

        f31 = self.F31(f21)
        f32 = self.F32(f22)
        f33 = self.F33(f23)


        return f31, f32, f33


###########Mutli UNET 2: UNET with multiple outputs####################################
class Multi2U_Net(nn.Module):
    def __init__(self,outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Multi2U_Net,self).__init__()
        
        #self.Con = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Conv1 = nn.Conv2d(outer_nc,32,kernel_size=4,stride=2,padding=1,bias=False)
        self.ReLU1 = nn.LeakyReLU(0.2,True)

        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv21 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.ReLU21 = nn.LeakyReLU(0.2,True)

        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv31 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        self.ReLU31 = nn.LeakyReLU(0.2,True)

        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv41 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.ReLU41 = nn.LeakyReLU(0.2,True)

        self.Conv5 = conv_block(ch_in=256,ch_out=256)
        self.Conv51 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.ReLU51 = nn.LeakyReLU(0.2,True)

        self.Conv6 = nn.Conv2d(256,256,kernel_size=4,stride=2,padding=1,bias=False)
        self.ReLU2 = nn.ReLU(inplace=True)
        self.Conv61 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.ReLU61 = nn.ReLU(inplace=True)

        self.Up5 = up_conv(ch_in=256,ch_out=256)
        self.Up51 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
        self.ReLU51up = nn.ReLU(inplace=True)

        self.Up4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.Bn1 = nn.BatchNorm2d(256)
        self.Dp1 = nn.Dropout(0.5,inplace=False)
        self.ReLU4 = nn.ReLU(inplace=True)
        self.Up41 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
        self.ReLU41up = nn.ReLU(inplace=True)

        self.Up3 = up_conv(ch_in=512,ch_out=128)
        self.Up31 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.ReLU31up = nn.ReLU(inplace=True)

        self.Up2 = up_conv(ch_in=256,ch_out=64)
        self.Up21 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        self.ReLU21up = nn.ReLU(inplace=True)

        self.Up11 = up_conv(ch_in=128,ch_out=32)
        self.Up12 = up_conv(ch_in=128,ch_out=32)

        self.F11 = nn.ConvTranspose2d(64,1,kernel_size=4,stride=2,padding=1,bias=True)
        self.ReLU31 = nn.LeakyReLU(0.8,True)
        self.F12 = nn.ConvTranspose2d(64,1,kernel_size=4,stride=2,padding=1,bias=True)
        self.ReLU32 = nn.LeakyReLU(0.8,True)

        self.F21 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        self.F22 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)

        self.F31 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        self.F32 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)

        self.C2onv1 = nn.Conv2d(2,32,kernel_size=4,stride=2,padding=1,bias=False)
        self.R2eLU1 = nn.LeakyReLU(0.2,True)

        self.C2onv2 = conv_block(ch_in=32,ch_out=64)
        self.C2onv21 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.R2eLU21 = nn.LeakyReLU(0.2,True)

        self.U2p2 = up_conv(ch_in=64,ch_out=32)

        self.G11 = nn.ConvTranspose2d(64,1,kernel_size=4,stride=2,padding=1,bias=True)
        self.GReLU = nn.LeakyReLU(0.8,True)

        self.G21 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        self.G22 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)



    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x1 = self.ReLU1(x1)

        x2 = self.Conv2(x1)
        x2 = self.Conv21(x2)
        x2 = self.ReLU21(x2)

        x3 = self.Conv3(x2)
        x3 = self.Conv31(x3)
        x3 = self.ReLU31(x3)

        x4 = self.Conv4(x3)
        x4 = self.Conv41(x4)
        x4 = self.ReLU41(x4)

        x5 = self.Conv5(x4)
        x5 = self.Conv51(x5)
        x5 = self.ReLU51(x5)


        x6 = self.Conv6(x5)
        x6 = self.ReLU2(x6)
        x6 = self.Conv61(x6)
        x6 = self.ReLU61(x6)

        # decoding + concat path
        d5 = self.Up5(x6)
        d5 = torch.cat((x5,d5),dim=1)
        d5 = self.Up51(d5)
        d5 = self.ReLU51up(d5)
        
        d4 = self.Up4(d5)
        d4 = self.Bn1(d4)
        d4 = self.Dp1(d4)
        d4 = self.ReLU4(d4)
        d4 = torch.cat((x4,d4),dim=1)
        d4 = self.Up41(d4)
        d4 = self.ReLU41up(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x3,d3),dim=1)
        d3 = self.Up31(d3)
        d3 = self.ReLU41up(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x2,d2),dim=1)
        d2 = self.Up21(d2)
        d2 = self.ReLU21up(d2)

        d11 = self.Up11(d2)
        d11 = torch.cat((x1,d11),dim=1)
        d12 = self.Up12(d2)
        d12 = torch.cat((x1,d12),dim=1)
        
        f11 = self.F11(d11)
        f11 = self.ReLU31(f11)
        f12 = self.F12(d12)
        f12 = self.ReLU32(f12)

        f21 = self.F21(f11)
        f22 = self.F22(f12)


        f31 = self.F31(f21)
        f32 = self.F32(f22)

        g00 = torch.cat((f31,f32),dim=1)
        g01 = self.C2onv1(g00)
        g01 = self.R2eLU1(g01)

        g02 = self.C2onv2(g01)
        g02 = self.C2onv21(g02)
        g02 = self.R2eLU21(g02)

        h00 = self.U2p2(g02)
        h00 = torch.cat((g01,h00),dim=1)

        h01 = self.G11(h00)
        h01 = self.GReLU(h01)

        h11 = self.G21(h01)
        h11 = self.G22(h11)


        return f31, f32, h11

####################UNET 3+################################################################

class UNet_3Plus(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        return d1


################Multi ASPP NET##########################################################
class MultiASPPU_Net(nn.Module):
    def __init__(self,outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(MultiASPPU_Net,self).__init__()
        
        #self.Con = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Conv1 = nn.Conv2d(outer_nc,32,kernel_size=4,stride=2,padding=1,bias=False)
        self.ReLU1 = nn.LeakyReLU(0.2,True)

        self.Conv2 = conv_block_sqex(ch_in=32,ch_out=64)

        self.Conv3 = conv_block_sqex(ch_in=64,ch_out=128)

        self.Conv4 = conv_block_sqex(ch_in=128,ch_out=256)

        self.Conv5 = conv_block_sqex(ch_in=256,ch_out=256)

        self.Conv6 = nn.Conv2d(256,256,kernel_size=4,stride=2,padding=1,bias=False)
        self.ReLU2 = nn.ReLU(inplace=True)

        self.ASPP = ASPP(256,256)
        #self.ReLU2 = nn.ReLU(inplace=True)

        self.Up5 = up_conv(ch_in=256,ch_out=256)

        self.Up4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.Bn1 = nn.BatchNorm2d(256)
        self.Dp1 = nn.Dropout(0.5,inplace=False)
        self.ReLU4 = nn.ReLU(inplace=True)

        self.Up3 = up_conv(ch_in=512,ch_out=128)

        self.Up2 = up_conv(ch_in=256,ch_out=64)

        self.Up11 = up_conv(ch_in=128,ch_out=32)

        self.F11 = nn.ConvTranspose2d(64,1,kernel_size=4,stride=2,padding=1,bias=True)
        self.ReLU31 = nn.LeakyReLU(0.8,True)

        self.F21 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        self.F31 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)

        self.Up12 = up_conv(ch_in=128,ch_out=32)

        self.F12 = nn.ConvTranspose2d(64,1,kernel_size=4,stride=2,padding=1,bias=True)
        self.ReLU32 = nn.LeakyReLU(0.8,True)

        self.F22 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        self.F32 = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)




    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x1 = self.ReLU1(x1)
        x2 = self.Conv2(x1)
        x3 = self.Conv3(x2)
        x4 = self.Conv4(x3)
        x5 = self.Conv5(x4)
        x6 = self.Conv6(x5)
        x6 = self.ReLU2(x6)

        aspp = self.ASPP(x6)        
        # decoding + concat path
        d5 = self.Up5(aspp)
        d5 = torch.cat((x5,d5),dim=1)
        
        d4 = self.Up4(d5)
        d4 = self.Bn1(d4)
        d4 = self.Dp1(d4)
        d4 = self.ReLU4(d4)
        d4 = torch.cat((x4,d4),dim=1)

        d3 = self.Up3(d4)
        d3 = torch.cat((x3,d3),dim=1)

        d2 = self.Up2(d3)
        d2 = torch.cat((x2,d2),dim=1)

        d11 = self.Up11(d2)
        d11 = torch.cat((x1,d11),dim=1)
        
        f11 = self.F11(d11)
        f11 = self.ReLU31(f11)

        f21 = self.F21(f11)
        f31 = self.F31(f21)

        d12 = self.Up12(d2)
        d12 = torch.cat((x1,d12),dim=1)
        
        f12 = self.F12(d12)
        f12 = self.ReLU32(f12)

        f22 = self.F22(f12)
        f32 = self.F32(f22)


        return f31, f32

#########################vgg 16############################################
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        #for x in range(16, 23):
        #    self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        #h = self.to_relu_4_3(h)
        #h_relu_4_3 = h
        out = h_relu_3_3
        return out
#######################UNET 2##############################################
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


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm)
        self.down = nn.MaxPool2d(2, 2, ceil_mode=True)
        #self.dropout = nn.Dropout2d(0.1)
        

    def forward(self, inputs):
        outputs = self.conv(inputs)
        #outputs = self.bn(outputs)
        #outputs = self.lr(outputs)
        outputs = self.down(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, True)
        # Transposed convolution
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset1 = (outputs2.size()[2]-inputs1.size()[2])
        offset2 = (outputs2.size()[3]-inputs1.size()[3])
        padding=[offset2//2,(offset2+1)//2,offset1//2,(offset1+1)//2]
        # Skip and concatenate 
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))

class NewU_Net(nn.Module):
    def __init__(self,outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(NewU_Net, self).__init__()
        self.is_deconv     = True
        self.in_channels   = outer_nc
        self.is_batchnorm  = False
        self.n_classes     = inner_nc
        
        filters = [64, 128, 256, 512, 1024]
        
        self.down1   = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2   = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3   = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4   = unetDown(filters[2], filters[3], self.is_batchnorm)
        self.center  = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.up4     = unetUp(filters[4], filters[3], self.is_deconv)
        self.up3     = unetUp(filters[3], filters[2], self.is_deconv)
        self.up2     = unetUp(filters[2], filters[1], self.is_deconv)
        self.up1     = unetUp(filters[1], filters[0], self.is_deconv)
        self.f1      = nn.Conv2d(filters[0],self.n_classes, 1)
        #self.final   = nn.ReLU(inplace=True)
        self.final   = nn.Conv2d(self.n_classes, self.n_classes, 1)
        
    def forward(self, inputs):
        label_dsp_dim = (101,101)
        down1  = self.down1(inputs)
        down2  = self.down2(down1)
        down3  = self.down3(down2)
        down4  = self.down4(down3)
        center = self.center(down4)
        up4    = self.up4(down4, center)
        up3    = self.up3(down3, up4)
        up2    = self.up2(down2, up3)
        up1    = self.up1(down1, up2)
        up1    = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        f1     = self.f1(up1)
        
        return self.final(f1)
    
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



class autoUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(autoUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, True)
        self.conv2 = unetConv2(out_size, out_size, True)
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
        return self.conv2(outputs3)
    
    
class autoUp2(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(autoUp2, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        self.conv2 = unetConv2(out_size, out_size, False)
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
    

class unetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv3, self).__init__()
        # Kernel size: 3*3, Stride: 1, Padding: 1
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.LeakyReLU(0.1))
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.LeakyReLU(0.1),
                                       nn.Dropout2d(0.01))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout2d(0.01))
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs
    
class autoUp3(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(autoUp3, self).__init__()
        self.conv = unetConv3(in_size, out_size, True)
        self.conv2 = unetConv3(out_size, out_size, True)
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

class Auto_Net(nn.Module):
    def __init__(self,outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Auto_Net, self).__init__()
        self.is_deconv     = False
        self.in_channels   = outer_nc
        self.is_batchnorm  = False
        self.n_classes     = inner_nc
        
        filters = [16, 32, 64, 128, 512]
        #filters = [2, 4, 8, 16, 32]
        
        latent_dim = 8

        self.down1   = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2   = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3   = unetDown(filters[1], filters[2], self.is_batchnorm)
        #self.down4   = unetDown(filters[2], filters[3], self.is_batchnorm)
        # self.center  = unetConv2(filters[3], filters[4], self.is_batchnorm)
        ##self.decoder_input1 = nn.Linear(filters[1]*250*51, latent_dim) #for marmousi 151x200
        #self.decoder_input1 = nn.Linear(filters[2]*125*26, latent_dim) #for marmousi 151x200
        #self.decoder_input = nn.Linear(latent_dim, filters[2]*500*102) #for marmousi 151x200
        self.decoder_input1 = nn.Linear(filters[2]*125*9, latent_dim) #for marmousi 101x101
        #self.decoder_input = nn.Linear(latent_dim, filters[3]*100*26) #for marmousi 101x101
        #self.decoder_input1 = nn.Linear(filters[1]*100*18, latent_dim) #for marmousi 101x101
        self.decoder_input = nn.Linear(latent_dim, filters[3]*50*13) #for marmousi 101x101
        
        
        #self.up4     = autoUp(filters[4], filters[3], self.is_deconv)
        self.up3     = autoUp(filters[3], filters[2], self.is_deconv)
        self.up2     = autoUp(filters[2], filters[1], self.is_deconv)
        self.up1     = autoUp(filters[1], filters[0], self.is_deconv)
        #self.upff1     = autoUp(filters[0], filters[0], self.is_deconv)
        #self.upff2     = autoUp(filters[0], filters[0], self.is_deconv)
        self.f1      =  nn.Conv2d(filters[0],self.n_classes, 1)
        #self.f2      =  nn.Conv2d(1,1,1)
        self.final   =  nn.Sigmoid()
        #self.final1  =  nn.Conv2d(1, 1, 1)
        
    def forward(self, inputs1, inputs2, lstart, epoch1, latentI, lowf):
        filters = [16, 32, 64, 128, 512]
        latent_dim = 8
        label_dsp_dim = (70,70)
        mintrue = torch.min(inputs1)
        maxtrue = torch.max(inputs1)
        
        print("input2 device :", inputs2.get_device())
        down1  = self.down1(inputs2)
        down2  = self.down2(down1)
        down3  = self.down3(down2)
        #down4  = self.down4(down3)
        
        #print("shape of down3 :", np.shape(down3))
        
        #print("shape of down2 :", np.shape(down2))
        result = torch.flatten(down3, start_dim=1)
        
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
        #latent1 = p
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
        z = z.view(-1, filters[3], 50, 13)
    
        up1    = self.up3(z)
        #print(" shape of up1 :", np.shape(up1))
        up1    = self.up2(up1)
        up1    = self.up1(up1)
        up1    = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        f1     = self.f1(up1)
        f1     = self.final(f1)
        #f1     = self.final1(f1)
        #f1     = self.final(f1)
        #f1     = f1/torch.max(f1)
        f1     = mintrue + f1*(maxtrue-mintrue)
        
        #f1     = mintrue + ((f1+1)*(maxtrue-mintrue)+1)/2
        
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
        #print("f1 device :", f1.get_device())
        #lossT = 0*f1
        if (epoch1 > lstart):
            [grad, lossT] = self.prop(inputs2, f1, lstart, epoch1, mintrue, maxtrue)
            grad = torch.unsqueeze(grad,0)
            grad = torch.unsqueeze(grad,0)
        #result = torch.flatten(f1, start_dim=1)
        #print(" shape of grad :", np.shape(grad))
        #print("lossT : ",lossT)
        return f1, grad
    
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
    def prop(self, inputs, vel, lstart, epoch1, mintrue, maxtrue):
        
        #torch.cuda.set_device(7)  #RB Necessary if device <> 0
        #GPU_string='cuda:'+str(7)
        #devicek = torch.device(GPU_string)
        net1out1 = vel*1000
        #net1out1 = vel*1000
        #net1out1 = (3550-1500)*vel+1500
        #print("---shape of vel---", str(np.shape(vel)))
        net1out1 = net1out1.detach()
        net1out1 = torch.squeeze(net1out1)
        #net1out1 = net1out1.to(devicek)
        devicek = net1out1.get_device()
        #net1out1[0:26,:] = 1500.0

        
        freq = 25
        dx = 15
        nt = 1000
        dt = 0.001
        num_shots = 5
        num_receivers_per_shot = 70
        num_sources_per_shot = 1
        num_dims = 2
        #ModelDim = [201,301]
        source_spacing = 70 * dx / num_shots
        receiver_spacing = 70 * dx / num_receivers_per_shot
        x_s = torch.zeros(num_shots, num_sources_per_shot, num_dims)
        x_s[:, 0, 1] = torch.arange(num_shots).float() * source_spacing
        x_r = torch.zeros(num_shots, num_receivers_per_shot, num_dims)
        x_r[0, :, 1] = torch.arange(
            num_receivers_per_shot).float() * receiver_spacing
        x_r[:, :, 1] = x_r[0, :, 1].repeat(num_shots, 1)

        source_amplitudes_true = (deepwave.wavelets.ricker(freq, nt, dt, 1/freq)
                                  .reshape(-1, 1, 1))
        #print("device ordinal :", self.devicek)
        source_amplitudes_true = source_amplitudes_true.to(devicek)
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
        mat2 = mat2 * mintrue * 1000
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
        
        receiver_amplitudes_true = receiver_amplitudes_true
        
        #print("receiver_amplitudes_true :", np.shape(receiver_amplitudes_true))
        #print("receiver_amplitudes_cte :", np.shape(receiver_amplitudes_cte))
        #receiver_amplitudes_true = receiver_amplitudes_true
        rcv_amps_true_max, _ = torch.abs(receiver_amplitudes_true).max(dim=0, keepdim=True)
        rcv_amps_true_norm = receiver_amplitudes_true / (rcv_amps_true_max.abs() + 1e-10)
        #rcv_amps_true_norm = receiver_amplitudes_true

        criterion1 = torch.nn.L1Loss()
        #vgg = Vgg16().type(torch.cuda.FloatTensor)
        #criterion2 = torch.nn.MSELoss()
        #print("shape of mat2 :", np.shape(mat2))
        

        if (epoch1 > lstart):
            net1out1.requires_grad = True
            optimizer2 = torch.optim.Adam([{'params': [net1out1], 'lr':10}])

        for epoch in range(num_epochs):
                #Shuffle shot coordinates
                idx = torch.randperm(num_shots)
                #idx = idx.type(torch.LongTensor)
                x_s = x_s.view(-1,2)[idx].view(x_s.size())
                #RB Shuffle true's seismograms sources with same random values
                rcv_amps_true_norm = rcv_amps_true_norm[:,idx,:]
                #RB Shuffle direct wave seismograms sources with the same random values
                receiver_amplitudes_cte = receiver_amplitudes_cte[:,idx,:]
        
                for it in range(num_batches):
                    #if (epoch1 > lstart):
                    optimizer2.zero_grad()
                    model2 = net1out1.clone()
                    model2 = torch.clamp(net1out1,min=mintrue*1000,max=maxtrue*1000)
                    #np.save('before108.npy',net1out1.cpu().detach().numpy())
                    #net1out1 = torch.clamp(net1out1,min=2000,max=4500)
                    prop = deepwave.scalar.Propagator({'vp': model2}, dx)
                    batch_src_amps = source_amplitudes_true.repeat(1, num_shots_per_batch, 1)
                    #print("shape of batch src amps")
                    #print(np.shape(batch_src_amps))
                    ############batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches].to(self.devicek)
                    batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches]
                    batch_rcv_amps_cte = receiver_amplitudes_cte[:,it::num_batches]
                    batch_x_s = x_s[it::num_batches].to(devicek)
                    ##################batch_x_s = x_s[it::num_batches]
                    #print("shape of batch src amps")
                    # print(np.shape(batch_x_s))
                    #####################batch_x_r = x_r[it::num_batches]
                    batch_x_r = x_r[it::num_batches].to(devicek)
                    #print("shape of batch receiver amps")
                    # print(np.shape(batch_x_r))
                    batch_rcv_amps_pred = prop(batch_src_amps, batch_x_s, batch_x_r, dt)
                    #print("batch_rcv_amps_pred")
                    #print(np.shape(batch_rcv_amps_pred))
                    batch_rcv_amps_pred = batch_rcv_amps_pred
                    batch_rcv_amps_pred_max, _ = torch.abs(batch_rcv_amps_pred).max(dim=0, keepdim=True)
                    # Normalize amplitudes by dividing by the maximum amplitude of each receiver
                    batch_rcv_amps_pred_norm = batch_rcv_amps_pred / (batch_rcv_amps_pred_max.abs() + 1e-10)
                    #batch_rcv_amps_pred_norm = batch_rcv_amps_pred 
                    ##############batch_rcv_amps_pred_norm = batch_rcv_amps_pred
                    
                    #print("shape of receiver amplitudes predicted")
                    # print(np.shape(batch_rcv_amps_pred))
                    lossinner1 = criterion1(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
                    #lossinner2 = criterion2(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
                    lossinner = lossinner1
                    #y_c_features = vgg(torch.unsqueeze(batch_rcv_amps_true,0))
                    #########model2.grad[0:26,:] = 0
                    #filen = './deepwave/epoch1'+str(epoch)+'.npy'
                    #np.save(filen,net1out1.cpu().detach().numpy())
                    ##############if (epoch == num_epochs-1):
                    ##########    sumlossinner += lossinner.item()
                    #########if (epoch1 > lstart):
                    lossinner.backward()
                    #net1out1.grad[(true[0,0,:,:]==2000)] = 0
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
                 
        return net1out1.grad
    

class AutoMarmousi_Net(nn.Module):
    def __init__(self,outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(AutoMarmousi_Net, self).__init__()
        self.is_deconv     = False
        self.in_channels   = outer_nc
        self.is_batchnorm  = False
        self.n_classes     = inner_nc
        
        filters = [16, 32, 64, 128, 512]
        #filters = [2, 4, 8, 16, 32]
        
        latent_dim = 8

        self.down1   = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2   = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3   = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4   = unetDown(filters[2], filters[3], self.is_batchnorm)
        # self.center  = unetConv2(filters[3], filters[4], self.is_batchnorm)
        ##self.decoder_input1 = nn.Linear(filters[1]*250*51, latent_dim) #for marmousi 151x200
        #self.decoder_input1 = nn.Linear(filters[2]*125*26, latent_dim) #for marmousi 151x200
        #self.decoder_input = nn.Linear(latent_dim, filters[2]*500*102) #for marmousi 151x200
        self.decoder_input1 = nn.Linear(filters[3]*69*43, latent_dim) #for marmousi 101x101
        #self.decoder_input = nn.Linear(latent_dim, filters[3]*100*26) #for marmousi 101x101
        #self.decoder_input1 = nn.Linear(filters[1]*100*18, latent_dim) #for marmousi 101x101
        self.decoder_input = nn.Linear(latent_dim, filters[3]*138*86) #for marmousi 101x101
        
        
        #self.up4     = autoUp(filters[4], filters[3], self.is_deconv)
        self.up3     = autoUp(filters[3], filters[2], self.is_deconv)
        self.up2     = autoUp(filters[2], filters[1], self.is_deconv)
        self.up1     = autoUp(filters[1], filters[0], self.is_deconv)
        #self.upff1     = autoUp(filters[0], filters[0], self.is_deconv)
        #self.upff2     = autoUp(filters[0], filters[0], self.is_deconv)
        self.f1      =  nn.Conv2d(filters[0],self.n_classes, 1)
        #self.f2      =  nn.Conv2d(1,1,1)
        #self.final   =  nn.Tanh()
        self.final  =  nn.Sigmoid()
        
    def forward(self, inputs1, inputs2, lstart, epoch1, latentI, lowf):
        filters = [16, 32, 64, 128, 512]
        latent_dim = 8
        label_dsp_dim = (1097,681)
        mintrue = torch.min(inputs1)
        maxtrue = torch.max(inputs1)
        print("shape of inputs2 :", np.shape(inputs2))
        down1  = self.down1(inputs2)
        down2  = self.down2(down1)
        down3  = self.down3(down2)
        down4  = self.down4(down3)
        
        print("shape of down4 :", np.shape(down4))
        
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
        z = z.view(-1, filters[3], 138, 86)
    
        up1    = self.up3(z)
        #print(" shape of up1 :", np.shape(up1))
        up1    = self.up2(up1)
        up1    = self.up1(up1)
        print("shape of up1 :", np.shape(up1))
        up1    = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        f1     = self.f1(up1)
        f1     = self.final(f1)
        #f1     = self.final1(f1)
        #f1     = self.final(f1)
        #f1     = f1/torch.max(f1)
        #print("shape of f1 :", np.shape(f1))
        
        f1    = mintrue + f1*(maxtrue-mintrue)
        #f1[(inputs1==1.5100)] = 1.510
        #f1     = lowf + f1
        #f1[(inputs1 == 1.510)] = 1.510
        #f1     = torch.clamp(f1,min=mintrue,max=maxtrue)
        
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
        #if (epoch1 > lstart):
        #    [grad, lossT] = self.prop(inputs2, f1, lstart, epoch1, mintrue, maxtrue, inputs1)
        #    grad = torch.unsqueeze(grad,0)
        #    grad = torch.unsqueeze(grad,0)
        #result = torch.flatten(f1, start_dim=1)
        #print(" shape of grad :", np.shape(grad))

        return f1, grad, latent1, lossT
    
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
        
        #torch.cuda.set_device(7)  #RB Necessary if device <> 0
        #GPU_string='cuda:'+str(7)
        #devicek = torch.device(GPU_string)
        #vel = vel.to(devicek)
        net1out1 = mintrue + vel*(maxtrue-mintrue)
        net1out1 = vel*1000
        #net1out1 = net1out2.to(devicek)
        #net1out1 = (3550-1500)*vel+1500
        #print("---shape of vel---", str(np.shape(vel)))
        #print("devicek :", devicek)
        net1out1 = net1out1.detach()
        net1out1 = torch.squeeze(net1out1)
        g1 = torch.arange(net1out1.size(dim=0))
        g1 = g1**2
        ss = g1.tile((200,1))
        ss = torch.transpose(ss,0,1)

        devicek = net1out1.get_device()
        #net1out1[0:26,:] = 1500.0

        
        freq = 14
        dx = 10
        nt = 4001
        dt = 0.001
        num_shots = 18
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
        x_r[0, :, 1] = torch.arange(
            num_receivers_per_shot).float() * receiver_spacing
        x_r[:, :, 1] = x_r[0, :, 1].repeat(num_shots, 1)

        source_amplitudes_true = (deepwave.wavelets.ricker(freq, nt, dt, 1/freq)
                                  .reshape(-1, 1, 1))
        #print("device ordinal :", self.devicek)
        source_amplitudes_true = source_amplitudes_true.to(devicek)
        #lstart = -1
        num_batches = 2
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
        mat2 = mat2 * 1510
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
        rcv_amps_true_norm = receiver_amplitudes_true / (rcv_amps_true_max.abs() + 1e-10)
        #rcv_amps_true_norm = receiver_amplitudes_true
        ss = ss.to(devicek)
        criterion1 = torch.nn.L1Loss()
        vgg = Vgg16().type(torch.cuda.FloatTensor)
        #criterion2 = torch.nn.MSELoss()
        #print("shape of mat2 :", np.shape(mat2))
        

        if (epoch1 > lstart):
            net1out1.requires_grad = True
            optimizer2 = torch.optim.Adam([{'params': [net1out1], 'lr':10}])

        for epoch in range(num_epochs):
                #Shuffle shot coordinates
                idx = torch.randperm(num_shots)
                #idx = idx.type(torch.LongTensor)
                x_s = x_s.view(-1,2)[idx].view(x_s.size())
                #RB Shuffle true's seismograms sources with same random values
                rcv_amps_true_norm = rcv_amps_true_norm[:,idx,:]
                #RB Shuffle direct wave seismograms sources with the same random values
                receiver_amplitudes_cte = receiver_amplitudes_cte[:,idx,:]
        
                for it in range(num_batches):
                    #if (epoch1 > lstart):
                    optimizer2.zero_grad()
                    model2 = net1out1.clone()
                    model2 = torch.clamp(net1out1,min=mintrue*1000,max=maxtrue*1000)
                    #np.save('before108.npy',net1out1.cpu().detach().numpy())
                    #net1out1 = torch.clamp(net1out1,min=2000,max=4500)
                    prop = deepwave.scalar.Propagator({'vp': model2}, dx)
                    batch_src_amps = source_amplitudes_true.repeat(1, num_shots_per_batch, 1)
                    #print("shape of batch src amps")
                    #print(np.shape(batch_src_amps))
                    ############batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches].to(self.devicek)
                    batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches]
                    batch_rcv_amps_cte = receiver_amplitudes_cte[:,it::num_batches]
                    batch_x_s = x_s[it::num_batches].to(devicek)
                    ##################batch_x_s = x_s[it::num_batches]
                    #print("shape of batch src amps")
                    # print(np.shape(batch_x_s))
                    #####################batch_x_r = x_r[it::num_batches]
                    batch_x_r = x_r[it::num_batches].to(devicek)
                    #print("shape of batch receiver amps")
                    # print(np.shape(batch_x_r))
                    batch_rcv_amps_pred = prop(batch_src_amps, batch_x_s, batch_x_r, dt)
                    #print("batch_rcv_amps_pred")
                    #print(np.shape(batch_rcv_amps_pred))
                    batch_rcv_amps_pred = batch_rcv_amps_pred - batch_rcv_amps_cte
                    batch_rcv_amps_pred_max, _ = torch.abs(batch_rcv_amps_pred).max(dim=0, keepdim=True)
                    # Normalize amplitudes by dividing by the maximum amplitude of each receiver
                    batch_rcv_amps_pred_norm = batch_rcv_amps_pred / (batch_rcv_amps_pred_max.abs() + 1e-10)
                    #batch_rcv_amps_pred_norm = batch_rcv_amps_pred 
                    ##############batch_rcv_amps_pred_norm = batch_rcv_amps_pred
                    
                    #print("shape of receiver amplitudes predicted")
                    # print(np.shape(batch_rcv_amps_pred))
                    lossinner1 = criterion1(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
                    
                    y_true1 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_true[:,0:3,:],0,1),0))
                    y_pred1 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_pred_norm[:,0:3,:],0,1),0))
                    y_true2 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_true[:,3:6,:],0,1),0))
                    y_pred2 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_pred_norm[:,3:6,:],0,1),0))
                    y_true3 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_true[:,6:9,:],0,1),0))
                    y_pred3 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_pred_norm[:,6:9,:],0,1),0))
                    
                    lossinner2 = criterion1(y_pred1,y_true1) + criterion1(y_pred2,y_true2) + criterion1(y_pred3,y_true3)
                    #lossinner2 = criterion2(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
                    lossinner = lossinner1 + lossinner2
                    
                    #y_c_features = vgg(torch.unsqueeze(batch_rcv_amps_true,0))
                    #########model2.grad[0:26,:] = 0
                    #filen = './deepwave/epoch1'+str(epoch)+'.npy'
                    #np.save(filen,net1out1.cpu().detach().numpy())
                    ##############if (epoch == num_epochs-1):
                    ##########    sumlossinner += lossinner.item()
                    #########if (epoch1 > lstart):
                    lossinner.backward()
                    #p1 = torch.rand(net1out1.grad.size()).to(devicek)
                    #p1 = p1*0.00005*(torch.max(net1out1.grad)-torch.min(net1out1.grad))
                    #net1out1.grad = p1 + net1out1.grad
                    net1out1.grad = net1out1.grad*ss
                    net1out1.grad[(true[0,0,:,:]==1.510)] = 0
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
                 
        return net1out1.grad, lossinner.item()
    
    
class VaeMarmousi3_Net(nn.Module):
    def __init__(self,outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(VaeMarmousi3_Net, self).__init__()
        self.is_deconv     = False
        self.in_channels   = outer_nc
        self.is_batchnorm  = False
        self.n_classes     = inner_nc
        
        filters = [16, 32, 64, 128, 512]
        #filters = [2, 4, 8, 16, 32]
        
        latent_dim = 8

        self.down1   = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2   = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3   = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4   = unetDown(filters[2], filters[3], self.is_batchnorm)
        # self.center  = unetConv2(filters[3], filters[4], self.is_batchnorm)
        ##self.decoder_input1 = nn.Linear(filters[1]*250*51, latent_dim) #for marmousi 151x200
        #self.decoder_input1 = nn.Linear(filters[2]*125*26, latent_dim) #for marmousi 151x200
        #self.decoder_input = nn.Linear(latent_dim, filters[2]*500*102) #for marmousi 151x200
        ########self.decoder_input1 = nn.Linear(filters[3]*63*13, latent_dim) #for marmousi 101x101 ######earlier
        self.smu = nn.Linear(filters[3]*63*13, latent_dim)
        self.svar = nn.Linear(filters[3]*63*13, latent_dim)
        #self.tanhl = nn.Tanh()
        #self.decoder_input = nn.Linear(latent_dim, filters[3]*100*26) #for marmousi 101x101
        #self.decoder_input1 = nn.Linear(filters[1]*100*18, latent_dim) #for marmousi 101x101
        self.decoder_input = nn.Linear(latent_dim, filters[3]*25*19) #for marmousi 101x101
        
        
        #self.up4     = autoUp(filters[4], filters[3], self.is_deconv)
        self.up3     = autoUp(filters[3], filters[2], self.is_deconv)
        self.up2     = autoUp(filters[2], filters[1], self.is_deconv)
        self.up1     = autoUp(filters[1], filters[0], self.is_deconv)
        #self.upff1     = autoUp(filters[0], filters[0], self.is_deconv)
        #self.upff2     = autoUp(filters[0], filters[0], self.is_deconv)
        self.f1      =  nn.Conv2d(filters[0],self.n_classes, 1)
        #self.f2      =  nn.Conv2d(1,1,1)
        self.final   =  nn.Sigmoid()
        #self.final1  =  nn.Conv2d(1, 1, 1)
        
    def forward(self, inputs1, inputs2, lstart, epoch1, latentI, lowf):
        filters = [16, 32, 64, 128, 512]
        latent_dim = 8
        label_dsp_dim = (151,200)
        mintrue = torch.min(inputs1)
        maxtrue = torch.max(inputs1)
        meandata = torch.mean(inputs2)
        stdata = torch.std(inputs2)
        down1  = self.down1(inputs2[:,:,1:4001:4,:])
        down2  = self.down2(down1)
        down3  = self.down3(down2)
        down4  = self.down4(down3)
        
        #print("shape of down4 :", np.shape(down4))
        
        #print("shape of down2 :", np.shape(down2))
        result = torch.flatten(down4, start_dim=1)
        
        #print("result shape :", np.shape(result))
        
        #p = self.decoder_input1(result)
        mu = self.smu(result)
        log_var = self.svar(result)
        p = self.reparameterize(mu, log_var)
        #p = self.tanhl(p)
        #down3  = self.down3(down2)
        #down4  = self.down4(down3)s
        #center = self.center(down4)
        #up4    = self.up4(center)
        #up3    = self.up3(up4)
        #up2    = self.up2(up3)
        #print("shape of down 4:", np.shape(down2))
        #print("shape of result:", np.shape(result))
        #latent1 = p
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
    
        up1    = self.up3(z)
        #print(" shape of up1 :", np.shape(up1))
        up1    = self.up2(up1)
        up1    = self.up1(up1)
        #print("shape of up1 :", np.shape(up1))
        up1    = up1[:,:,1:1+label_dsp_dim[0],0:1+label_dsp_dim[1]].contiguous()
        f1     = self.f1(up1)
        f1     = self.final(f1)
        #f1     = self.final1(f1)
        #f1     = self.final(f1)
        #f1     = f1/torch.max(f1)
        #print("shape of f1 :", np.shape(f1))
        
        f1    = mintrue + f1*(maxtrue-mintrue)
        f1[(inputs1==1.500)] = 1.500
        #f1     = lowf + f1
        #f1[(inputs1 == 1.510)] = 1.510
        #f1     = torch.clamp(f1,min=mintrue,max=maxtrue)
        
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
            [grad, lossT] = self.prop(inputs2, f1, lstart, epoch1, mintrue, maxtrue, inputs1)
            grad = grad.to(inputs2.get_device())
            grad = torch.unsqueeze(grad,0)
            grad = torch.unsqueeze(grad,0)
        #result = torch.flatten(f1, start_dim=1)
        #print(" shape of grad :", np.shape(grad))

        return f1, grad, mu, log_var, lossT
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    
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
        
        torch.cuda.set_device(4)  #RB Necessary if device <> 0
        GPU_string='cuda:'+str(4)
        devicek = torch.device(GPU_string)
        #torch.cuda.set_device(7)  #RB Necessary if device <> 0
        #GPU_string='cuda:'+str(7)
        #devicek = torch.device(GPU_string)
        #vel = vel.to(devicek)
        #net1out1 = mintrue + vel*(maxtrue-mintrue)
        net1out1 = vel*1000
        #net1out1 = net1out2.to(devicek)
        #net1out1 = (3550-1500)*vel+1500
        #print("---shape of vel---", str(np.shape(vel)))
        #print("devicek :", devicek)
        net1out1 = net1out1.detach()
        net1out1 = torch.squeeze(net1out1)
        g1 = torch.arange(net1out1.size(dim=0))
        g1 = g1**2
        ss = g1.tile((200,1))
        ss = torch.transpose(ss,0,1)

        net1out1 = net1out1.to(devicek)
        #devicek = net1out1.get_device()
        #net1out1[0:26,:] = 1500.0

        
        freq = 8
        dx = 10
        nt = 4001
        dt = 0.001
        num_shots = 18
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
        x_r[0, :, 1] = torch.arange(
            num_receivers_per_shot).float() * receiver_spacing
        x_r[:, :, 1] = x_r[0, :, 1].repeat(num_shots, 1)

        source_amplitudes_true = (deepwave.wavelets.ricker(freq, nt, dt, 1/freq)
                                  .reshape(-1, 1, 1))
        #print("device ordinal :", self.devicek)
        source_amplitudes_true = source_amplitudes_true.to(devicek)
        #lstart = -1
        num_batches = 2
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
        rcv_amps_true_norm = receiver_amplitudes_true / (rcv_amps_true_max.abs() + 1e-10)
        #rcv_amps_true_norm = receiver_amplitudes_true
        ss = ss.to(devicek)
        criterion1 = torch.nn.L1Loss()
        #vgg = Vgg16().type(torch.cuda.FloatTensor)
        #criterion2 = torch.nn.MSELoss()
        #print("shape of mat2 :", np.shape(mat2))
        

        if (epoch1 > lstart):
            net1out1.requires_grad = True
            optimizer2 = torch.optim.Adam([{'params': [net1out1], 'lr':10}])

        for epoch in range(num_epochs):
                #Shuffle shot coordinates
                idx = torch.randperm(num_shots)
                #idx = idx.type(torch.LongTensor)
                x_s = x_s.view(-1,2)[idx].view(x_s.size())
                #RB Shuffle true's seismograms sources with same random values
                rcv_amps_true_norm = rcv_amps_true_norm[:,idx,:]
                #RB Shuffle direct wave seismograms sources with the same random values
                receiver_amplitudes_cte = receiver_amplitudes_cte[:,idx,:]
        
                for it in range(num_batches):
                    #if (epoch1 > lstart):
                    optimizer2.zero_grad()
                    model2 = net1out1.clone()
                    model2 = torch.clamp(net1out1,min=mintrue*1000,max=maxtrue*1000)
                    #np.save('before108.npy',net1out1.cpu().detach().numpy())
                    #net1out1 = torch.clamp(net1out1,min=2000,max=4500)
                    prop = deepwave.scalar.Propagator({'vp': model2}, dx)
                    batch_src_amps = source_amplitudes_true.repeat(1, num_shots_per_batch, 1)
                    #print("shape of batch src amps")
                    #print(np.shape(batch_src_amps))
                    ############batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches].to(self.devicek)
                    batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches]
                    batch_rcv_amps_cte = receiver_amplitudes_cte[:,it::num_batches]
                    batch_x_s = x_s[it::num_batches].to(devicek)
                    ##################batch_x_s = x_s[it::num_batches]
                    #print("shape of batch src amps")
                    # print(np.shape(batch_x_s))
                    #####################batch_x_r = x_r[it::num_batches]
                    batch_x_r = x_r[it::num_batches].to(devicek)
                    #print("shape of batch receiver amps")
                    # print(np.shape(batch_x_r))
                    batch_rcv_amps_pred = prop(batch_src_amps, batch_x_s, batch_x_r, dt)
                    #print("batch_rcv_amps_pred")
                    #print(np.shape(batch_rcv_amps_pred))
                    batch_rcv_amps_pred = batch_rcv_amps_pred - batch_rcv_amps_cte
                    batch_rcv_amps_pred_max, _ = torch.abs(batch_rcv_amps_pred).max(dim=0, keepdim=True)
                    # Normalize amplitudes by dividing by the maximum amplitude of each receiver
                    batch_rcv_amps_pred_norm = batch_rcv_amps_pred / (batch_rcv_amps_pred_max.abs() + 1e-10)
                    #batch_rcv_amps_pred_norm = batch_rcv_amps_pred 
                    ##############batch_rcv_amps_pred_norm = batch_rcv_amps_pred
                    
                    #print("shape of receiver amplitudes predicted")
                    # print(np.shape(batch_rcv_amps_pred))
                    lossinner1 = criterion1(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
                    
                    #y_true1 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_true[:,0:3,:],0,1),0))
                    #y_pred1 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_pred_norm[:,0:3,:],0,1),0))
                    #y_true2 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_true[:,3:6,:],0,1),0))
                    #y_pred2 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_pred_norm[:,3:6,:],0,1),0))
                    #y_true3 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_true[:,6:9,:],0,1),0))
                    #y_pred3 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_pred_norm[:,6:9,:],0,1),0))
                    
                    #lossinner2 = criterion1(y_pred1,y_true1) + criterion1(y_pred2,y_true2) + criterion1(y_pred3,y_true3)
                    #lossinner2 = criterion2(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
                    lossinner = lossinner1
                    
                    #y_c_features = vgg(torch.unsqueeze(batch_rcv_amps_true,0))
                    #########model2.grad[0:26,:] = 0
                    #filen = './deepwave/epoch1'+str(epoch)+'.npy'
                    #np.save(filen,net1out1.cpu().detach().numpy())
                    ##############if (epoch == num_epochs-1):
                    ##########    sumlossinner += lossinner.item()
                    #########if (epoch1 > lstart):
                    lossinner.backward()
                    #p1 = torch.rand(net1out1.grad.size()).to(devicek)
                    #p1 = p1*0.00005*(torch.max(net1out1.grad)-torch.min(net1out1.grad))
                    #net1out1.grad = p1 + net1out1.grad
                    net1out1.grad = net1out1.grad*ss
                    net1out1.grad[(true[0,0,:,:]==1.500)] = 0
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
                 
        return net1out1.grad, lossinner.item()
    
    
class VaeMarmousi_Net(nn.Module):
    def __init__(self,outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(VaeMarmousi_Net, self).__init__()
        self.is_deconv     = False
        self.in_channels   = outer_nc
        self.is_batchnorm  = False
        self.n_classes     = inner_nc
        
        filters = [16, 32, 64, 128, 512]
        #filters = [2, 4, 8, 16, 32]
        
        latent_dim = 8

        self.down1   = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2   = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3   = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4   = unetDown(filters[2], filters[3], self.is_batchnorm)
        # self.center  = unetConv2(filters[3], filters[4], self.is_batchnorm)
        ##self.decoder_input1 = nn.Linear(filters[1]*250*51, latent_dim) #for marmousi 151x200
        #self.decoder_input1 = nn.Linear(filters[2]*125*26, latent_dim) #for marmousi 151x200
        #self.decoder_input = nn.Linear(latent_dim, filters[2]*500*102) #for marmousi 151x200
        ########self.decoder_input1 = nn.Linear(filters[3]*63*13, latent_dim) #for marmousi 101x101 ######earlier
        self.smu = nn.Linear(filters[3]*63*13, latent_dim)
        self.svar = nn.Linear(filters[3]*63*13, latent_dim)
        #self.decoder_input = nn.Linear(latent_dim, filters[3]*100*26) #for marmousi 101x101
        #self.decoder_input1 = nn.Linear(filters[1]*100*18, latent_dim) #for marmousi 101x101
        self.decoder_input = nn.Linear(latent_dim, filters[3]*25*13) #for marmousi 101x101
        
        
        #self.up4     = autoUp(filters[4], filters[3], self.is_deconv)
        self.up3     = autoUp(filters[3], filters[2], self.is_deconv)
        self.up2     = autoUp(filters[2], filters[1], self.is_deconv)
        self.up1     = autoUp(filters[1], filters[0], self.is_deconv)
        #self.upff1     = autoUp(filters[0], filters[0], self.is_deconv)
        #self.upff2     = autoUp(filters[0], filters[0], self.is_deconv)
        self.f1      =  nn.Conv2d(filters[0],self.n_classes, 1)
        #self.f2      =  nn.Conv2d(1,1,1)
        self.final   =  nn.Sigmoid()
        #self.final1  =  nn.Conv2d(1, 1, 1)
        
    def forward(self, inputs1, inputs2, lstart, epoch1, latentI, lowf):
        filters = [16, 32, 64, 128, 512]
        latent_dim = 8
        label_dsp_dim = (100,200)
        mintrue = torch.min(inputs1)
        maxtrue = torch.max(inputs1)
        down1  = self.down1(inputs2[:,:,1:4001:4,:])
        down2  = self.down2(down1)
        down3  = self.down3(down2)
        down4  = self.down4(down3)
        
        #print("shape of down4 :", np.shape(down4))
        
        #print("shape of down2 :", np.shape(down2))
        result = torch.flatten(down4, start_dim=1)
        
        #print("result shape :", np.shape(result))
        
        #p = self.decoder_input1(result)
        mu = self.smu(result)
        log_var = self.svar(result)
        p = self.reparameterize(mu, log_var)
        
        #down3  = self.down3(down2)
        #down4  = self.down4(down3)s
        #center = self.center(down4)
        #up4    = self.up4(center)
        #up3    = self.up3(up4)
        #up2    = self.up2(up3)
        #print("shape of down 4:", np.shape(down2))
        #print("shape of result:", np.shape(result))
        #latent1 = p
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
        z = z.view(-1, filters[3], 13, 25)
    
        up1    = self.up3(z)
        #print(" shape of up1 :", np.shape(up1))
        up1    = self.up2(up1)
        up1    = self.up1(up1)
        #print("shape of up1 :", np.shape(up1))
        up1    = up1[:,:,1:1+label_dsp_dim[0],0:1+label_dsp_dim[1]].contiguous()
        f1     = self.f1(up1)
        f1     = self.final(f1)
        #f1     = self.final1(f1)
        #f1     = self.final(f1)
        #f1     = f1/torch.max(f1)
        #print("shape of f1 :", np.shape(f1))
        
        f1    = mintrue + f1*(maxtrue-mintrue)
        f1[(inputs1==1.5100)] = 1.510
        #f1     = lowf + f1
        #f1[(inputs1 == 1.510)] = 1.510
        #f1     = torch.clamp(f1,min=mintrue,max=maxtrue)
        
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
            [grad, lossT] = self.prop(inputs2, f1, lstart, epoch1, mintrue, maxtrue, inputs1)
            grad = torch.unsqueeze(grad,0)
            grad = torch.unsqueeze(grad,0)
        #result = torch.flatten(f1, start_dim=1)
        #print(" shape of grad :", np.shape(grad))

        return f1, grad, mu, log_var, lossT
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    
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
        
        #torch.cuda.set_device(7)  #RB Necessary if device <> 0
        #GPU_string='cuda:'+str(7)
        #devicek = torch.device(GPU_string)
        #vel = vel.to(devicek)
        #net1out1 = mintrue + vel*(maxtrue-mintrue)
        net1out1 = vel*1000
        #net1out1 = net1out2.to(devicek)
        #net1out1 = (3550-1500)*vel+1500
        #print("---shape of vel---", str(np.shape(vel)))
        #print("devicek :", devicek)
        net1out1 = net1out1.detach()
        net1out1 = torch.squeeze(net1out1)
        g1 = torch.arange(net1out1.size(dim=0))
        g1 = g1**2
        ss = g1.tile((200,1))
        ss = torch.transpose(ss,0,1)

        devicek = net1out1.get_device()
        #net1out1[0:26,:] = 1500.0

        
        freq = 14
        dx = 10
        nt = 4001
        dt = 0.001
        num_shots = 18
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
        x_r[0, :, 1] = torch.arange(
            num_receivers_per_shot).float() * receiver_spacing
        x_r[:, :, 1] = x_r[0, :, 1].repeat(num_shots, 1)

        source_amplitudes_true = (deepwave.wavelets.ricker(freq, nt, dt, 1/freq)
                                  .reshape(-1, 1, 1))
        #print("device ordinal :", self.devicek)
        source_amplitudes_true = source_amplitudes_true.to(devicek)
        #lstart = -1
        num_batches = 2
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
        mat2 = mat2 * 1510
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
        rcv_amps_true_norm = receiver_amplitudes_true / (rcv_amps_true_max.abs() + 1e-10)
        #rcv_amps_true_norm = receiver_amplitudes_true
        ss = ss.to(devicek)
        criterion1 = torch.nn.L1Loss()
        #vgg = Vgg16().type(torch.cuda.FloatTensor)
        #criterion2 = torch.nn.MSELoss()
        #print("shape of mat2 :", np.shape(mat2))
        

        if (epoch1 > lstart):
            net1out1.requires_grad = True
            optimizer2 = torch.optim.Adam([{'params': [net1out1], 'lr':10}])

        for epoch in range(num_epochs):
                #Shuffle shot coordinates
                idx = torch.randperm(num_shots)
                #idx = idx.type(torch.LongTensor)
                x_s = x_s.view(-1,2)[idx].view(x_s.size())
                #RB Shuffle true's seismograms sources with same random values
                rcv_amps_true_norm = rcv_amps_true_norm[:,idx,:]
                #RB Shuffle direct wave seismograms sources with the same random values
                receiver_amplitudes_cte = receiver_amplitudes_cte[:,idx,:]
        
                for it in range(num_batches):
                    #if (epoch1 > lstart):
                    optimizer2.zero_grad()
                    model2 = net1out1.clone()
                    model2 = torch.clamp(net1out1,min=mintrue*1000,max=maxtrue*1000)
                    #np.save('before108.npy',net1out1.cpu().detach().numpy())
                    #net1out1 = torch.clamp(net1out1,min=2000,max=4500)
                    prop = deepwave.scalar.Propagator({'vp': model2}, dx)
                    batch_src_amps = source_amplitudes_true.repeat(1, num_shots_per_batch, 1)
                    #print("shape of batch src amps")
                    #print(np.shape(batch_src_amps))
                    ############batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches].to(self.devicek)
                    batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches]
                    batch_rcv_amps_cte = receiver_amplitudes_cte[:,it::num_batches]
                    batch_x_s = x_s[it::num_batches].to(devicek)
                    ##################batch_x_s = x_s[it::num_batches]
                    #print("shape of batch src amps")
                    # print(np.shape(batch_x_s))
                    #####################batch_x_r = x_r[it::num_batches]
                    batch_x_r = x_r[it::num_batches].to(devicek)
                    #print("shape of batch receiver amps")
                    # print(np.shape(batch_x_r))
                    batch_rcv_amps_pred = prop(batch_src_amps, batch_x_s, batch_x_r, dt)
                    #print("batch_rcv_amps_pred")
                    #print(np.shape(batch_rcv_amps_pred))
                    batch_rcv_amps_pred = batch_rcv_amps_pred - batch_rcv_amps_cte
                    batch_rcv_amps_pred_max, _ = torch.abs(batch_rcv_amps_pred).max(dim=0, keepdim=True)
                    # Normalize amplitudes by dividing by the maximum amplitude of each receiver
                    batch_rcv_amps_pred_norm = batch_rcv_amps_pred / (batch_rcv_amps_pred_max.abs() + 1e-10)
                    #batch_rcv_amps_pred_norm = batch_rcv_amps_pred 
                    ##############batch_rcv_amps_pred_norm = batch_rcv_amps_pred
                    
                    #print("shape of receiver amplitudes predicted")
                    # print(np.shape(batch_rcv_amps_pred))
                    lossinner1 = criterion1(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
                    
                    #y_true1 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_true[:,0:3,:],0,1),0))
                    #y_pred1 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_pred_norm[:,0:3,:],0,1),0))
                    #y_true2 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_true[:,3:6,:],0,1),0))
                    #y_pred2 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_pred_norm[:,3:6,:],0,1),0))
                    #y_true3 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_true[:,6:9,:],0,1),0))
                    #y_pred3 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_pred_norm[:,6:9,:],0,1),0))
                    
                    #lossinner2 = criterion1(y_pred1,y_true1) + criterion1(y_pred2,y_true2) + criterion1(y_pred3,y_true3)
                    #lossinner2 = criterion2(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
                    lossinner = lossinner1
                    
                    #y_c_features = vgg(torch.unsqueeze(batch_rcv_amps_true,0))
                    #########model2.grad[0:26,:] = 0
                    #filen = './deepwave/epoch1'+str(epoch)+'.npy'
                    #np.save(filen,net1out1.cpu().detach().numpy())
                    ##############if (epoch == num_epochs-1):
                    ##########    sumlossinner += lossinner.item()
                    #########if (epoch1 > lstart):
                    lossinner.backward()
                    #p1 = torch.rand(net1out1.grad.size()).to(devicek)
                    #p1 = p1*0.00005*(torch.max(net1out1.grad)-torch.min(net1out1.grad))
                    #net1out1.grad = p1 + net1out1.grad
                    net1out1.grad = net1out1.grad*ss
                    net1out1.grad[(true[0,0,:,:]==1.510)] = 0
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
                 
        return net1out1.grad, lossinner.item()
    
    
class AutoMarmousi21_Net(nn.Module):
    def __init__(self,outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(AutoMarmousi21_Net, self).__init__()
        self.is_deconv     = False
        self.in_channels   = outer_nc
        self.is_batchnorm  = False
        self.n_classes     = inner_nc
        
        filters = [16, 32, 64, 128, 512]
        #filters = [2, 4, 8, 16, 32]
        
        latent_dim = 8

        self.down1   = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2   = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3   = unetDown(filters[1], filters[2], self.is_batchnorm)
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
        self.up3     = autoUp3(filters[3], filters[2], self.is_deconv)
        self.up2     = autoUp3(filters[2], filters[1], self.is_deconv)
        self.up1     = autoUp3(filters[1], filters[0], self.is_deconv)
        #self.upff1     = autoUp(filters[0], filters[0], self.is_deconv)
        #self.upff2     = autoUp(filters[0], filters[0], self.is_deconv)
        self.f1      =  nn.Conv2d(filters[0],self.n_classes, 1)
        #self.f2      =  nn.Conv2d(1,1,1)
        self.final   =  nn.Sigmoid()
        #self.final1  =  nn.Conv2d(1, 1, 1)
        
    def forward(self, inputs1, inputs2, lstart, epoch1, latentI, lowf):
        filters = [16, 32, 64, 128, 512]
        latent_dim = 8
        label_dsp_dim = (151,200)
        mintrue = torch.min(inputs1)
        maxtrue = torch.max(inputs1)
        meandata = torch.mean(inputs2)
        stddata = torch.std(inputs2)
        down1  = self.down1((inputs2[:,:,1:4001:4,:]-meandata)/stddata)
        down2  = self.down2(down1)
        down3  = self.down3(down2)
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
        f1[(inputs1==1.500)] = 1.500
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
            [grad, lossT] = self.prop(inputs2, f1, lstart, epoch1, mintrue, maxtrue, inputs1)
            grad = grad.to(inputs2.get_device())
            grad = torch.unsqueeze(grad,0)
            grad = torch.unsqueeze(grad,0)
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
        net1out1 = vel*1000
        #net1out1 = net1out2.to(devicek)
        #net1out1 = (3550-1500)*vel+1500
        #print("---shape of vel---", str(np.shape(vel)))
        #print("devicek :", devicek)
        net1out1 = net1out1.detach()
        net1out1 = torch.squeeze(net1out1)
        g1 = torch.arange(net1out1.size(dim=0))
        g1 = g1**2
        ss = g1.tile((200,1))
        ss = torch.transpose(ss,0,1)
        net1out1 = net1out1.to(devicek)
        #devicek = net1out1.get_device()
        #net1out1[0:26,:] = 1500.0

        
        freq = 8
        dx = 10
        nt = 4001
        dt = 0.001
        num_shots = 18
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
        x_r[0, :, 1] = torch.arange(
            num_receivers_per_shot).float() * receiver_spacing
        x_r[:, :, 1] = x_r[0, :, 1].repeat(num_shots, 1)

        source_amplitudes_true = (deepwave.wavelets.ricker(freq, nt, dt, 1/freq)
                                  .reshape(-1, 1, 1))
        #print("device ordinal :", self.devicek)
        source_amplitudes_true = source_amplitudes_true.to(devicek)
        #lstart = -1
        num_batches = 2
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
        rcv_amps_true_norm = receiver_amplitudes_true / (rcv_amps_true_max.abs() + 1e-10)
        #rcv_amps_true_norm = receiver_amplitudes_true

        criterion1 = torch.nn.L1Loss()
        #vgg = Vgg16().type(torch.cuda.FloatTensor)
        #criterion2 = torch.nn.MSELoss()
        #print("shape of mat2 :", np.shape(mat2))
        

        if (epoch1 > lstart):
            net1out1.requires_grad = True
            optimizer2 = torch.optim.Adam([{'params': [net1out1], 'lr':10}])

        for epoch in range(num_epochs):
                #Shuffle shot coordinates
                idx = torch.randperm(num_shots)
                #idx = idx.type(torch.LongTensor)
                x_s = x_s.view(-1,2)[idx].view(x_s.size())
                #RB Shuffle true's seismograms sources with same random values
                rcv_amps_true_norm = rcv_amps_true_norm[:,idx,:]
                #RB Shuffle direct wave seismograms sources with the same random values
                receiver_amplitudes_cte = receiver_amplitudes_cte[:,idx,:]
        
                for it in range(num_batches):
                    #if (epoch1 > lstart):
                    optimizer2.zero_grad()
                    model2 = net1out1.clone()
                    model2 = torch.clamp(net1out1,min=mintrue*1000,max=maxtrue*1000)
                    #np.save('before108.npy',net1out1.cpu().detach().numpy())
                    #net1out1 = torch.clamp(net1out1,min=2000,max=4500)
                    prop = deepwave.scalar.Propagator({'vp': model2}, dx)
                    batch_src_amps = source_amplitudes_true.repeat(1, num_shots_per_batch, 1)
                    #print("shape of batch src amps")
                    #print(np.shape(batch_src_amps))
                    ############batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches].to(self.devicek)
                    batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches]
                    batch_rcv_amps_cte = receiver_amplitudes_cte[:,it::num_batches]
                    batch_x_s = x_s[it::num_batches].to(devicek)
                    ##################batch_x_s = x_s[it::num_batches]
                    #print("shape of batch src amps")
                    # print(np.shape(batch_x_s))
                    #####################batch_x_r = x_r[it::num_batches]
                    batch_x_r = x_r[it::num_batches].to(devicek)
                    #print("shape of batch receiver amps")
                    # print(np.shape(batch_x_r))
                    batch_rcv_amps_pred = prop(batch_src_amps, batch_x_s, batch_x_r, dt)
                    #print("batch_rcv_amps_pred")
                    #print(np.shape(batch_rcv_amps_pred))
                    batch_rcv_amps_pred = batch_rcv_amps_pred - batch_rcv_amps_cte
                    batch_rcv_amps_pred_max, _ = torch.abs(batch_rcv_amps_pred).max(dim=0, keepdim=True)
                    # Normalize amplitudes by dividing by the maximum amplitude of each receiver
                    batch_rcv_amps_pred_norm = batch_rcv_amps_pred / (batch_rcv_amps_pred_max.abs() + 1e-10)
                    #batch_rcv_amps_pred_norm = batch_rcv_amps_pred 
                    ##############batch_rcv_amps_pred_norm = batch_rcv_amps_pred
                    
                    #print("shape of receiver amplitudes predicted")
                    # print(np.shape(batch_rcv_amps_pred))
                    lossinner1 = criterion1(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
                    #lossinner2 = criterion2(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
                    lossinner = lossinner1
                    #y_c_features = vgg(torch.unsqueeze(batch_rcv_amps_true,0))
                    #########model2.grad[0:26,:] = 0
                    #filen = './deepwave/epoch1'+str(epoch)+'.npy'
                    #np.save(filen,net1out1.cpu().detach().numpy())
                    ##############if (epoch == num_epochs-1):
                    ##########    sumlossinner += lossinner.item()
                    #########if (epoch1 > lstart):
                    lossinner.backward()
                    net1out1.grad = net1out1.grad * ss
                    net1out1.grad[(true[0,0,:,:]==1500)] = 0
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
                 
        return net1out1.grad, lossinner.item()
    
    
class AutoMarmousi22_Net(nn.Module):
    def __init__(self,outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(AutoMarmousi22_Net, self).__init__()
        self.is_deconv     = False
        self.in_channels   = outer_nc
        self.is_batchnorm  = True
        self.n_classes     = inner_nc
        
        filters = [16, 32, 64, 128, 512]
        #filters = [2, 4, 8, 16, 32]
        
        latent_dim = 8

        self.combine = nn.Conv2d(self.in_channels,1,1)
        self.down1   = unetDown(1, filters[0], self.is_batchnorm)
        self.down2   = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3   = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4   = unetDown(filters[2], filters[3], self.is_batchnorm)
        # self.center  = unetConv2(filters[3], filters[4], self.is_batchnorm)
        ##self.decoder_input1 = nn.Linear(filters[1]*250*51, latent_dim) #for marmousi 151x200
        #self.decoder_input1 = nn.Linear(filters[2]*125*26, latent_dim) #for marmousi 151x200
        #self.decoder_input = nn.Linear(latent_dim, filters[2]*500*102) #for marmousi 151x200
        self.decoder_input1 = nn.Linear(filters[3]*63*13, latent_dim) #for marmousi 101x101
        #self.tanhl = nn.LeakyReLU(0.1)
        #self.decoder_input = nn.Linear(latent_dim, filters[3]*100*26) #for marmousi 101x101
        #self.decoder_input1 = nn.Linear(filters[1]*100*18, latent_dim) #for marmousi 101x101
        self.decoder_input = nn.Linear(latent_dim, filters[3]*25*13) #for marmousi 101x101
        
        
        #self.up4     = autoUp(filters[4], filters[3], self.is_deconv)
        self.up3     = autoUp(filters[3], filters[2], self.is_deconv)
        self.up2     = autoUp(filters[2], filters[1], self.is_deconv)
        self.up1     = autoUp(filters[1], filters[0], self.is_deconv)
        #self.upff1     = autoUp(filters[0], filters[0], self.is_deconv)
        #self.upff2     = autoUp(filters[0], filters[0], self.is_deconv)
        self.f1      =  nn.Conv2d(filters[0],self.n_classes, 1)
        #self.f2      =  nn.Conv2d(1,1,1)
        self.final   =  nn.Sigmoid()
        #self.final1  =  nn.Conv2d(1, 1, 1)
        
    def forward(self, inputs1, inputs2, lstart, epoch1, latentI, lowf):
        filters = [16, 32, 64, 128, 512]
        latent_dim = 8
        label_dsp_dim = (100,200)
        mintrue = torch.min(inputs1)
        maxtrue = torch.max(inputs1)
        comb = self.combine(inputs2[:,:,1:4001:4,:])
        down1  = self.down1(comb)
        down2  = self.down2(down1)
        down3  = self.down3(down2)
        down4  = self.down4(down3)
        
        #print("shape of down3 :", np.shape(down3))
        #print("shape of down4 :", np.shape(down4))
        
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
            
        # p = self.tanhl(p)
        #p = torch.randn([1,1,1,8])
        #z = 0.5*torch.ones([1,1,1,64])
        z = self.decoder_input(p)
        #z = z.view(-1, filters[3], 250, 51) #for marmousi model
        z = z.view(-1, filters[3], 13, 25)
    
        up1    = self.up3(z)
        #print(" shape of up1 :", np.shape(up1))
        up1    = self.up2(up1)
        up1    = self.up1(up1)
        #print("shape of up1 :", np.shape(up1))
        up1    = up1[:,:,1:1+label_dsp_dim[0],0:1+label_dsp_dim[1]].contiguous()
        f1     = self.f1(up1)
        f1     = self.final(f1)
        #f1     = self.final1(f1)
        #f1     = self.final(f1)
        #f1     = f1/torch.max(f1)
        #print("shape of f1 :", np.shape(f1))
        
        f1    = mintrue + f1*(maxtrue-mintrue)
        f1[(inputs1==1.5100)] = 1.510
        #f1     = lowf + f1
        #f1[(inputs1 == 1.510)] = 1.510
        #f1     = torch.clamp(f1,min=mintrue,max=maxtrue)
        
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
            [grad, lossT] = self.prop(inputs2, f1, lstart, epoch1, mintrue, maxtrue, inputs1)
            grad = torch.unsqueeze(grad,0)
            grad = torch.unsqueeze(grad,0)
        #result = torch.flatten(f1, start_dim=1)
        #print(" shape of grad :", np.shape(grad))

        return f1, grad, latent1, lossT
    
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
        
        #torch.cuda.set_device(7)  #RB Necessary if device <> 0
        #GPU_string='cuda:'+str(7)
        #devicek = torch.device(GPU_string)
        #vel = vel.to(devicek)
        #net1out1 = mintrue + vel*(maxtrue-mintrue)
        net1out1 = vel*1000
        #net1out1 = net1out2.to(devicek)
        #net1out1 = (3550-1500)*vel+1500
        #print("---shape of vel---", str(np.shape(vel)))
        #print("devicek :", devicek)
        net1out1 = net1out1.detach()
        net1out1 = torch.squeeze(net1out1)
        g1 = torch.arange(net1out1.size(dim=0))
        g1 = g1**2.5
        ss = g1.tile((200,1))
        ss = torch.transpose(ss,0,1)

        devicek = net1out1.get_device()
        #net1out1[0:26,:] = 1500.0

        
        freq = 14
        dx = 10
        nt = 4001
        dt = 0.001
        num_shots = 18
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
        x_r[0, :, 1] = torch.arange(
            num_receivers_per_shot).float() * receiver_spacing
        x_r[:, :, 1] = x_r[0, :, 1].repeat(num_shots, 1)

        source_amplitudes_true = (deepwave.wavelets.ricker(freq, nt, dt, 1/freq)
                                  .reshape(-1, 1, 1))
        #print("device ordinal :", self.devicek)
        source_amplitudes_true = source_amplitudes_true.to(devicek)
        #lstart = -1
        num_batches = 2
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
        
        #print("shape of ss :", ss.size())
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
        mat2 = mat2 * 1510
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
        vgg = Vgg16().type(torch.cuda.FloatTensor)
        criterion2 = torch.nn.MSELoss()
        #print("shape of mat2 :", np.shape(mat2))
        

        if (epoch1 > lstart):
            net1out1.requires_grad = True
            optimizer2 = torch.optim.Adam([{'params': [net1out1], 'lr':10}])

        for epoch in range(num_epochs):
                #Shuffle shot coordinates
                idx = torch.randperm(num_shots)
                #idx = idx.type(torch.LongTensor)
                x_s = x_s.view(-1,2)[idx].view(x_s.size())
                #RB Shuffle true's seismograms sources with same random values
                rcv_amps_true_norm = rcv_amps_true_norm[:,idx,:]
                #RB Shuffle direct wave seismograms sources with the same random values
                receiver_amplitudes_cte = receiver_amplitudes_cte[:,idx,:]
        
                for it in range(num_batches):
                    #if (epoch1 > lstart):
                    optimizer2.zero_grad()
                    model2 = net1out1.clone()
                    model2 = torch.clamp(net1out1,min=mintrue*1000,max=maxtrue*1000)
                    #np.save('before108.npy',net1out1.cpu().detach().numpy())
                    #net1out1 = torch.clamp(net1out1,min=2000,max=4500)
                    prop = deepwave.scalar.Propagator({'vp': model2}, dx)
                    batch_src_amps = source_amplitudes_true.repeat(1, num_shots_per_batch, 1)
                    #print("shape of batch src amps")
                    #print(np.shape(batch_src_amps))
                    ############batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches].to(self.devicek)
                    batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches]
                    batch_rcv_amps_cte = receiver_amplitudes_cte[:,it::num_batches]
                    batch_x_s = x_s[it::num_batches].to(devicek)
                    ##################batch_x_s = x_s[it::num_batches]
                    #print("shape of batch src amps")
                    # print(np.shape(batch_x_s))
                    #####################batch_x_r = x_r[it::num_batches]
                    batch_x_r = x_r[it::num_batches].to(devicek)
                    #print("shape of batch receiver amps")
                    # print(np.shape(batch_x_r))
                    batch_rcv_amps_pred = prop(batch_src_amps, batch_x_s, batch_x_r, dt)
                    #print("batch_rcv_amps_pred")
                    #print(np.shape(batch_rcv_amps_pred))
                    batch_rcv_amps_pred = batch_rcv_amps_pred - batch_rcv_amps_cte
                    batch_rcv_amps_pred_max, _ = torch.abs(batch_rcv_amps_pred).max(dim=0, keepdim=True)
                    # Normalize amplitudes by dividing by the maximum amplitude of each receiver
                    #batch_rcv_amps_pred_norm = batch_rcv_amps_pred / (batch_rcv_amps_pred_max.abs() + 1e-10)
                    batch_rcv_amps_pred_norm = batch_rcv_amps_pred 
                    ##############batch_rcv_amps_pred_norm = batch_rcv_amps_pred
                    
                    #print("shape of receiver amplitudes predicted")
                    # print(np.shape(batch_rcv_amps_pred))
                    lossinner1 = criterion1(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
                    #lossinner2 = criterion2(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
                    #y_true1 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_true[:,0:3,:],0,1),0))
                    #y_pred1 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_pred_norm[:,0:3,:],0,1),0))
                    #y_true2 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_true[:,3:6,:],0,1),0))
                    #y_pred2 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_pred_norm[:,3:6,:],0,1),0))
                    #y_true3 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_true[:,6:9,:],0,1),0))
                    #y_pred3 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_pred_norm[:,6:9,:],0,1),0))
                    
                    #lossinner2 = criterion1(y_pred1,y_true1) + criterion1(y_pred2,y_true2) + criterion1(y_pred3,y_true3)
                    lossinner = lossinner1
                    
                    ####y_c_features = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_true[:,0:3,:],0,1),0))
                    #########model2.grad[0:26,:] = 0
                    #filen = './deepwave/epoch1'+str(epoch)+'.npy'
                    #np.save(filen,net1out1.cpu().detach().numpy())
                    ##############if (epoch == num_epochs-1):
                    ##########    sumlossinner += lossinner.item()
                    #########if (epoch1 > lstart):
                    lossinner.backward()
                    net1out1.grad = net1out1.grad*ss
                    net1out1.grad[(true[0,0,:,:]==1.510)] = 0
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
                 
        return net1out1.grad, lossinner.item()
    
    
class AutoMarmousi23_Net(nn.Module):
    def __init__(self,outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(AutoMarmousi23_Net, self).__init__()
        self.is_deconv     = False
        self.in_channels   = outer_nc
        self.is_batchnorm  = False
        self.n_classes     = inner_nc
        
        filters = [16, 32, 64, 128, 512]
        #filters = [2, 4, 8, 16, 32]
        
        latent_dim = 8

        self.down1   = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2   = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3   = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4   = unetDown(filters[2], filters[3], self.is_batchnorm)
        # self.center  = unetConv2(filters[3], filters[4], self.is_batchnorm)
        ##self.decoder_input1 = nn.Linear(filters[1]*250*51, latent_dim) #for marmousi 151x200
        #self.decoder_input1 = nn.Linear(filters[2]*125*26, latent_dim) #for marmousi 151x200
        #self.decoder_input = nn.Linear(latent_dim, filters[2]*500*102) #for marmousi 151x200
        self.decoder_input1 = nn.Linear(filters[3]*63*13, latent_dim) #for marmousi 101x101
        #self.decoder_input = nn.Linear(latent_dim, filters[3]*100*26) #for marmousi 101x101
        #self.decoder_input1 = nn.Linear(filters[1]*100*18, latent_dim) #for marmousi 101x101
        self.decoder_input = nn.Linear(latent_dim, filters[3]*26*13) #for marmousi 101x101
        
        
        #self.up4     = autoUp(filters[4], filters[3], self.is_deconv)
        self.up3     = autoUp(filters[3], filters[2], self.is_deconv)
        self.up2     = autoUp(filters[2], filters[1], self.is_deconv)
        self.up1     = autoUp(filters[1], filters[0], self.is_deconv)
        #self.upff1     = autoUp(filters[0], filters[0], self.is_deconv)
        #self.upff2     = autoUp(filters[0], filters[0], self.is_deconv)
        self.f1      =  nn.Conv2d(filters[0],self.n_classes, 1)
        #self.f2      =  nn.Conv2d(1,1,1)
        self.final   =  nn.Sigmoid()
        #self.final1  =  nn.Conv2d(1, 1, 1)
        
    def forward(self, inputs1, inputs2, lstart, epoch1, latentI, lowf):
        filters = [16, 32, 64, 128, 512]
        latent_dim = 8
        label_dsp_dim = (100,200)
        mintrue = torch.min(inputs1)
        maxtrue = torch.max(inputs1)
        mindata = torch.min(inputs2)
        maxdata = torch.max(inputs2)
        print("shapes of inputs2 :", np.shape(inputs2))
        down1  = self.down1(2*(inputs2[:,1:30:2,1:4001:4,:]-mindata)/(maxdata-mindata)-1)
        down2  = self.down2(down1)
        down3  = self.down3(down2)
        down4  = self.down4(down3)
        
        #print("shape of down3 :", np.shape(down3))
        #print("shape of down4 :", np.shape(down4))
        
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
        z = z.view(-1, filters[3], 13, 26)
    
        up1    = self.up3(z)
        print(" shape of up3 :", np.shape(up1))
        up1    = self.up2(up1)
        print(" shape of up2 :", np.shape(up1))
        up1    = self.up1(up1)
        print("shape of up1 :", np.shape(up1))
        up1    = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        f1     = self.f1(up1)
        f1     = self.final(f1)
        #f1     = self.final1(f1)
        #f1     = self.final(f1)
        #f1     = f1/torch.max(f1)
        #print("shape of f1 :", np.shape(f1))
        
        f1    = mintrue + f1*(maxtrue-mintrue)
        f1[(inputs1==1.5100)] = 1.510
        #f1     = lowf + f1
        #f1[(inputs1 == 1.510)] = 1.510
        #f1     = torch.clamp(f1,min=mintrue,max=maxtrue)
        
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
            [grad, lossT] = self.prop(inputs2, f1, lstart, epoch1, mintrue, maxtrue, inputs1)
            grad = torch.unsqueeze(grad,0)
            grad = torch.unsqueeze(grad,0)
        #result = torch.flatten(f1, start_dim=1)
        #print(" shape of grad :", np.shape(grad))

        return f1, grad, latent1, lossT
    
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
        
        #torch.cuda.set_device(7)  #RB Necessary if device <> 0
        #GPU_string='cuda:'+str(7)
        #devicek = torch.device(GPU_string)
        #vel = vel.to(devicek)
        #net1out1 = mintrue + vel*(maxtrue-mintrue)
        net1out1 = vel*1000
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

        devicek = net1out1.get_device()
        #net1out1[0:26,:] = 1500.0

        
        freq = 14
        dx = 10
        nt = 4001
        dt = 0.001
        num_shots = 27
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
        x_r[0, :, 1] = torch.arange(
            num_receivers_per_shot).float() * receiver_spacing
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
        
        #print("shape of ss :", ss.size())
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
        mat2 = mat2 * 1510
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
        rcv_amps_true_norm = receiver_amplitudes_true / (rcv_amps_true_max.abs() + 1e-10)
        #rcv_amps_true_norm = receiver_amplitudes_true

        criterion1 = torch.nn.L1Loss()
        #vgg = Vgg16().type(torch.cuda.FloatTensor)
        #criterion2 = torch.nn.MSELoss()
        #print("shape of mat2 :", np.shape(mat2))
        

        if (epoch1 > lstart):
            net1out1.requires_grad = True
            optimizer2 = torch.optim.Adam([{'params': [net1out1], 'lr':10}])

        for epoch in range(num_epochs):
                #Shuffle shot coordinates
                idx = torch.randperm(num_shots)
                #idx = idx.type(torch.LongTensor)
                x_s = x_s.view(-1,2)[idx].view(x_s.size())
                #RB Shuffle true's seismograms sources with same random values
                rcv_amps_true_norm = rcv_amps_true_norm[:,idx,:]
                #RB Shuffle direct wave seismograms sources with the same random values
                receiver_amplitudes_cte = receiver_amplitudes_cte[:,idx,:]
        
                for it in range(num_batches):
                    #if (epoch1 > lstart):
                    optimizer2.zero_grad()
                    model2 = net1out1.clone()
                    model2 = torch.clamp(net1out1,min=mintrue*1000,max=maxtrue*1000)
                    #np.save('before108.npy',net1out1.cpu().detach().numpy())
                    #net1out1 = torch.clamp(net1out1,min=2000,max=4500)
                    prop = deepwave.scalar.Propagator({'vp': model2}, dx)
                    batch_src_amps = source_amplitudes_true.repeat(1, num_shots_per_batch, 1)
                    #print("shape of batch src amps")
                    #print(np.shape(batch_src_amps))
                    ############batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches].to(self.devicek)
                    batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches]
                    batch_rcv_amps_cte = receiver_amplitudes_cte[:,it::num_batches]
                    batch_x_s = x_s[it::num_batches].to(devicek)
                    ##################batch_x_s = x_s[it::num_batches]
                    #print("shape of batch src amps")
                    # print(np.shape(batch_x_s))
                    #####################batch_x_r = x_r[it::num_batches]
                    batch_x_r = x_r[it::num_batches].to(devicek)
                    #print("shape of batch receiver amps")
                    # print(np.shape(batch_x_r))
                    batch_rcv_amps_pred = prop(batch_src_amps, batch_x_s, batch_x_r, dt)
                    #print("batch_rcv_amps_pred")
                    #print(np.shape(batch_rcv_amps_pred))
                    batch_rcv_amps_pred = batch_rcv_amps_pred - batch_rcv_amps_cte
                    batch_rcv_amps_pred_max, _ = torch.abs(batch_rcv_amps_pred).max(dim=0, keepdim=True)
                    # Normalize amplitudes by dividing by the maximum amplitude of each receiver
                    batch_rcv_amps_pred_norm = batch_rcv_amps_pred / (batch_rcv_amps_pred_max.abs() + 1e-10)
                    #batch_rcv_amps_pred_norm = batch_rcv_amps_pred 
                    ##############batch_rcv_amps_pred_norm = batch_rcv_amps_pred
                    
                    #print("shape of receiver amplitudes predicted")
                    # print(np.shape(batch_rcv_amps_pred))
                    lossinner1 = criterion1(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
                    #lossinner2 = criterion2(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
                    #y_true1 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_true[:,0:3,:],0,1),0))
                    #y_pred1 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_pred_norm[:,0:3,:],0,1),0))
                    #y_true2 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_true[:,3:6,:],0,1),0))
                    #y_pred2 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_pred_norm[:,3:6,:],0,1),0))
                    #y_true3 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_true[:,6:9,:],0,1),0))
                    #y_pred3 = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_pred_norm[:,6:9,:],0,1),0))
                    
                    #lossinner2 = criterion1(y_pred1,y_true1) + criterion1(y_pred2,y_true2) + criterion1(y_pred3,y_true3)
                    lossinner = lossinner1
                    
                    ####y_c_features = vgg(torch.unsqueeze(torch.swapaxes(batch_rcv_amps_true[:,0:3,:],0,1),0))
                    #########model2.grad[0:26,:] = 0
                    #filen = './deepwave/epoch1'+str(epoch)+'.npy'
                    #np.save(filen,net1out1.cpu().detach().numpy())
                    ##############if (epoch == num_epochs-1):
                    ##########    sumlossinner += lossinner.item()
                    #########if (epoch1 > lstart):
                    lossinner.backward()
                    net1out1.grad = net1out1.grad*ss
                    net1out1.grad[(true[0,0,:,:]==1.510)] = 0
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
                 
        return net1out1.grad, lossinner.item()
    
    
class AutoN_Net(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(AutoN_Net, self).__init__()
        self.is_deconv = False
        self.in_channels = outer_nc
        self.is_batchnorm = False
        self.n_classes = inner_nc

        filters = [16, 32, 64, 128, 512]
        latent_dim = 8

        #self.conv1 = nn.Conv2d(self.in_channels, 4, 3, 1, 1, 1)
        self.down1 = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2 = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3 = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4 = unetDown(filters[2], filters[3], self.is_batchnorm)
        #self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.decoder_input1 = nn.Linear(filters[3]*63*5,latent_dim)
        

        self.decoder_input = nn.Linear(latent_dim, filters[3]*9*9)

        #self.up4 = autoUp(filters[3], filters[3], self.is_deconv)
        self.up3 = autoUp(filters[3], filters[2], self.is_deconv)
        self.up2 = autoUp(filters[2], filters[1], self.is_deconv)
        self.up1 = autoUp(filters[1], filters[0], self.is_deconv)
        self.f1 = nn.Conv2d(filters[0], self.n_classes, 1)
        self.final = nn.Tanh()
        

    def forward(self, inputs1, inputs2, lstart, epoch1, p, lowf):
        filters = [16, 32, 64, 128, 512]
        latent_dim = 8
        label_dsp_dim = (70,70)
        mintrue = torch.min(inputs1)
        maxtrue = torch.max(inputs1)
        
        inputs2 = (inputs2 - torch.mean(inputs2))/torch.std(inputs2)
        #print("input2 device :", inputs2.get_device())
        #cc1 = self.conv1(inputs2)
        down1  = self.down1(inputs2)
        down2  = self.down2(down1)
        down3  = self.down3(down2)
        down4  = self.down4(down3)
        
        #killprint("shape of down4 :", np.shape(down4))
        
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
        #latent1 = p
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
        z = z.view(-1, filters[3], 9, 9)
    
        up1    = self.up3(z)
        #print(" shape of up1 :", np.shape(up1))
        up1    = self.up2(up1)
        up1    = self.up1(up1)
        #print(" shape of up1 :", np.shape(up1))
        up1    = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        f1     = self.f1(up1)
        f1     = self.final(f1)
        #f1     = self.final1(f1)
        #f1     = self.final(f1)
        #f1     = f1/torch.max(f1)
        #f1     = mintrue + f1*(maxtrue-mintrue)
        f1      = lowf + f1
        f1      = torch.clamp(f1, min=mintrue,max=maxtrue)
        #f1     = mintrue + ((f1+1)*(maxtrue-mintrue)+1)/2
        
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
        #print("f1 device :", f1.get_device())
        #lossT = 0*f1
        if (epoch1 > lstart):
            grad = self.prop(inputs2, f1, lstart, epoch1, mintrue, maxtrue)
            grad = torch.unsqueeze(grad,0)
            grad = torch.unsqueeze(grad,0)
        #result = torch.flatten(f1, start_dim=1)
        #print(" shape of grad :", np.shape(grad))
        #print("lossT : ",lossT)
        return f1, grad

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
    def prop(self, inputs, vel, lstart, epoch1, mintrue, maxtrue):
        
        #torch.cuda.set_device(7)  #RB Necessary if device <> 0
        #GPU_string='cuda:'+str(7)
        #devicek = torch.device(GPU_string)
        net1out1 = vel*1000
        #net1out1 = vel*1000
        #net1out1 = (3550-1500)*vel+1500
        #print("---shape of vel---", str(np.shape(vel)))
        net1out1 = net1out1.detach()
        net1out1 = torch.squeeze(net1out1)
        #net1out1 = net1out1.to(devicek)
        devicek = net1out1.get_device()
        #net1out1[0:26,:] = 1500.0

        
        freq = 25
        dx = 15
        nt = 1000
        dt = 0.001
        num_shots = 6
        num_receivers_per_shot = 70
        num_sources_per_shot = 1
        num_dims = 2
        ny = 70
        #ModelDim = [201,301]
        source_spacing = 70 * dx / num_shots
        receiver_spacing = 70 * dx / num_receivers_per_shot
        x_s = torch.zeros(num_shots, num_sources_per_shot, num_dims)
        #x_s[:, 0, 1] = torch.arange(num_shots).float() * source_spacing
        x_s[:, 0, 1] = torch.linspace(0,(ny-1)*dx,num_shots)
        x_r = torch.zeros(num_shots, num_receivers_per_shot, num_dims)
        x_r[0, :, 1] = torch.arange(
            num_receivers_per_shot).float() * receiver_spacing
        x_r[:, :, 1] = x_r[0, :, 1].repeat(num_shots, 1)

        source_amplitudes_true = (deepwave.wavelets.ricker(freq, nt, dt, 1/freq)
                                  .reshape(-1, 1, 1))
        #print("device ordinal :", self.devicek)
        source_amplitudes_true = source_amplitudes_true.to(devicek)
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
        mat2 = mat2 * mintrue * 1000
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
        
        receiver_amplitudes_true = receiver_amplitudes_true
        
        #print("receiver_amplitudes_true :", np.shape(receiver_amplitudes_true))
        #print("receiver_amplitudes_cte :", np.shape(receiver_amplitudes_cte))
        #receiver_amplitudes_true = receiver_amplitudes_true
        rcv_amps_true_max, _ = torch.abs(receiver_amplitudes_true).max(dim=0, keepdim=True)
        rcv_amps_true_norm = receiver_amplitudes_true / (rcv_amps_true_max.abs() + 1e-10)
        #rcv_amps_true_norm = receiver_amplitudes_true

        criterion1 = torch.nn.L1Loss()
        #vgg = Vgg16().type(torch.cuda.FloatTensor)
        #criterion2 = torch.nn.MSELoss()
        #print("shape of mat2 :", np.shape(mat2))
        

        if (epoch1 > lstart):
            net1out1.requires_grad = True
            #net1out1.retain_grad()
            optimizer2 = torch.optim.Adam([{'params': [net1out1], 'lr':10}])

        for epoch in range(num_epochs):
                #Shuffle shot coordinates
                idx = torch.randperm(num_shots)
                #idx = idx.type(torch.LongTensor)
                x_s = x_s.view(-1,2)[idx].view(x_s.size())
                #RB Shuffle true's seismograms sources with same random values
                rcv_amps_true_norm = rcv_amps_true_norm[:,idx,:]
                #RB Shuffle direct wave seismograms sources with the same random values
                receiver_amplitudes_cte = receiver_amplitudes_cte[:,idx,:]
        
                for it in range(num_batches):
                    #if (epoch1 > lstart):
                    optimizer2.zero_grad()
                    model2 = net1out1.clone()
                    model2 = torch.clamp(net1out1,min=mintrue*1000,max=maxtrue*1000)
                    #np.save('before108.npy',net1out1.cpu().detach().numpy())
                    #net1out1 = torch.clamp(net1out1,min=2000,max=4500)
                    prop = deepwave.scalar.Propagator({'vp': model2}, dx)
                    batch_src_amps = source_amplitudes_true.repeat(1, num_shots_per_batch, 1)
                    #print("shape of batch src amps")
                    #print(np.shape(batch_src_amps))
                    ############batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches].to(self.devicek)
                    batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches]
                    batch_rcv_amps_cte = receiver_amplitudes_cte[:,it::num_batches]
                    batch_x_s = x_s[it::num_batches].to(devicek)
                    ##################batch_x_s = x_s[it::num_batches]
                    #print("shape of batch src amps")
                    # print(np.shape(batch_x_s))
                    #####################batch_x_r = x_r[it::num_batches]
                    batch_x_r = x_r[it::num_batches].to(devicek)
                    #print("shape of batch receiver amps")
                    # print(np.shape(batch_x_r))
                    batch_rcv_amps_pred = prop(batch_src_amps, batch_x_s, batch_x_r, dt)
                    #print("batch_rcv_amps_pred")
                    #print(np.shape(batch_rcv_amps_pred))
                    batch_rcv_amps_pred = batch_rcv_amps_pred
                    batch_rcv_amps_pred_max, _ = torch.abs(batch_rcv_amps_pred).max(dim=0, keepdim=True)
                    # Normalize amplitudes by dividing by the maximum amplitude of each receiver
                    batch_rcv_amps_pred_norm = batch_rcv_amps_pred / (batch_rcv_amps_pred_max.abs() + 1e-10)
                    #batch_rcv_amps_pred_norm = batch_rcv_amps_pred 
                    ##############batch_rcv_amps_pred_norm = batch_rcv_amps_pred
                    
                    #print("shape of receiver amplitudes predicted")
                    # print(np.shape(batch_rcv_amps_pred))
                    lossinner1 = criterion1(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
                    #lossinner2 = torch.sqrt(criterion2(batch_rcv_amps_pred_norm, batch_rcv_amps_true))
                    lossinner = lossinner1
                    #y_c_features = vgg(torch.unsqueeze(batch_rcv_amps_true,0))
                    #########model2.grad[0:26,:] = 0
                    #filen = './deepwave/epoch1'+str(epoch)+'.npy'
                    #np.save(filen,net1out1.cpu().detach().numpy())
                    ##############if (epoch == num_epochs-1):
                    ##########    sumlossinner += lossinner.item()
                    #########if (epoch1 > lstart):
                    lossinner.backward()
                    #net1out1.grad[(true[0,0,:,:]==2000)] = 0
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
                 
        return net1out1.grad
    # def sample(self,
    #            num_samples:int,
    #            current_device: int, **kwargs) -> Tensor:
    #     """
    #     Samples from the latent space and return the corresponding
    #     image space map.
    #     :param num_samples: (Int) Number of samples
    #     :param current_device: (Int) Device to run the model
    #     :return: (Tensor)
    #     """
    #     z = torch.randn(num_samples,
    #                     self.latent_dim)

    #     z = z.to(current_device)

    #     samples = self.decode(z)
    #     return samples

class Vae_Net(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Vae_Net, self).__init__()
        self.is_deconv = False
        self.in_channels = outer_nc
        self.is_batchnorm = False
        self.n_classes = inner_nc

        filters = [16, 32, 64, 128, 512]
        latent_dim = 8

        self.down1 = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2 = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3 = unetDown(filters[1], filters[2], self.is_batchnorm)
        #self.down4 = unetDown(filters[2], filters[3], self.is_batchnorm)
        #self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.fc_mu = nn.Linear(filters[2]*125*9, latent_dim)
        self.fc_var = nn.Linear(filters[2]*125*9, latent_dim)
        

        self.decoder_input = nn.Linear(latent_dim, filters[3]*50*13)

        #self.up4 = autoUp(filters[3], filters[3], self.is_deconv)
        self.up3 = autoUp(filters[3], filters[2], self.is_deconv)
        self.up2 = autoUp(filters[2], filters[1], self.is_deconv)
        self.up1 = autoUp(filters[1], filters[0], self.is_deconv)
        self.f1 = nn.Conv2d(filters[0], self.n_classes, 1)
        self.final = nn.ReLU(inplace=True)

    def encode(self, inputs):
        label_dsp_dim = (70,70)
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        #down4 = self.down4(down3)
        #center = self.center(down4)
        
        #print("shape of down")
        #print(np.shape(down4))

        result = torch.flatten(down3, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        #center = self.center(down4)

        #print("shape of down4")
        #print(np.shape(down4))
        return [mu, log_var]

    def decode(self, inputs):
        filters = [16, 32, 64, 128, 512]
        label_dsp_dim = (70,70)
        decoder_input = self.decoder_input(inputs)
        decoder_input = decoder_input.view(-1, filters[3], 50, 13)
        #up4 = self.up4(decoder_input)
        up3 = self.up3(decoder_input)
        up2 = self.up2(up3)
        up1 = self.up1(up2)
        up1 = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        f1  = self.f1(up1)
        return self.final(f1)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
        

    # def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
    #     """
    #     Reparameterization trick to sample from N(mu, var) from
    #     N(0,1).
    #     :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    #     :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    #     :return: (Tensor) [B x D]
    #     """
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return eps * std + mu

    def forward(self, inputs1, inputs2, lstart, epoch1, p, lowf):
        mu,log_var = self.encode(inputs2)
        z = self.reparameterize(mu, log_var)
        #print("shape of z: ", np.shape(z))
        de1 = self.decode(z)
        de1 = 2000 + de1*(4500-2000)
        de1[(inputs1==2000)] = 2000
        
        #de1 = torch.clip(de1, min=2000, max= 4500)
        #print("shape of de1 :", np.shape(de1))
        #print(type(de1))
        grad = 0*de1
        if (epoch1 > lstart):
            grad = self.prop(inputs2, de1, lstart, epoch1, inputs1)
            grad = torch.unsqueeze(grad,0)
            grad = torch.unsqueeze(grad,0)
            #print("shape of de2")
            #print(np.shape(de2))    
        return  de1, mu, log_var, grad

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
    def prop(self, inputs, vel, lstart, epoch1, true):
        
        #torch.cuda.set_device(7)  #RB Necessary if device <> 0
        #GPU_string='cuda:'+str(7)
        #devicek = torch.device(GPU_string)
        net1out1 = vel
        #net1out1 = vel
        #net1out1 = (3550-1500)*vel+1500
        #print("---shape of vel---", str(np.shape(vel)))
        net1out1 = net1out1.detach()
        net1out1 = torch.squeeze(net1out1)
        #net1out1 = net1out1.to(devicek)
        devicek = net1out1.get_device()
        #net1out1[0:26,:] = 1500.0

        
        freq = 30
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
        source_amplitudes_true = source_amplitudes_true.to(devicek)
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
        min1 = torch.min(net1out1)
        #print("min1 :", min1)
        #print(min1.get_device())
        #min1 = min1.to(self.device1)
        mat2 = torch.ones(net1out1.size()[0],net1out1.size()[1]).to(devicek)
        mat2 = mat2 * 2000.0
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
        
        receiver_amplitudes_true = receiver_amplitudes_true-receiver_amplitudes_cte
        
        #print("receiver_amplitudes_true :", np.shape(receiver_amplitudes_true))
        #print("receiver_amplitudes_cte :", np.shape(receiver_amplitudes_cte))
        #receiver_amplitudes_true = receiver_amplitudes_true
        rcv_amps_true_max, _ = torch.abs(receiver_amplitudes_true).max(dim=0, keepdim=True)
        rcv_amps_true_norm = receiver_amplitudes_true / (rcv_amps_true_max.abs() + 1e-10)

        criterion = torch.nn.MSELoss()

        #print("shape of mat2 :", np.shape(mat2))
        

        if (epoch1 > lstart):
            net1out1.requires_grad = True
            optimizer2 = torch.optim.Adam([{'params': [net1out1], 'lr':10}])

        for epoch in range(num_epochs):
                #Shuffle shot coordinates
                idx = torch.randperm(num_shots)
                #idx = idx.type(torch.LongTensor)
                x_s = x_s.view(-1,2)[idx].view(x_s.size())
                #RB Shuffle true's seismograms sources with same random values
                rcv_amps_true_norm = rcv_amps_true_norm[:,idx,:]
                #RB Shuffle direct wave seismograms sources with the same random values
                receiver_amplitudes_cte = receiver_amplitudes_cte[:,idx,:]
        
                for it in range(num_batches):
                    #if (epoch1 > lstart):
                    optimizer2.zero_grad()
                    model2 = net1out1.clone()
                    model2 = torch.clamp(net1out1,min=2000,max=4500)
                    #np.save('before108.npy',net1out1.cpu().detach().numpy())
                    #net1out1 = torch.clamp(net1out1,min=2000,max=4500)
                    prop = deepwave.scalar.Propagator({'vp': model2}, dx)
                    batch_src_amps = source_amplitudes_true.repeat(1, num_shots_per_batch, 1)
                    #print("shape of batch src amps")
                    #print(np.shape(batch_src_amps))
                    ############batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches].to(self.devicek)
                    batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches]
                    batch_rcv_amps_cte = receiver_amplitudes_cte[:,it::num_batches]
                    batch_x_s = x_s[it::num_batches].to(devicek)
                    ##################batch_x_s = x_s[it::num_batches]
                    #print("shape of batch src amps")
                    # print(np.shape(batch_x_s))
                    #####################batch_x_r = x_r[it::num_batches]
                    batch_x_r = x_r[it::num_batches].to(devicek)
                    #print("shape of batch receiver amps")
                    # print(np.shape(batch_x_r))
                    batch_rcv_amps_pred = prop(batch_src_amps, batch_x_s, batch_x_r, dt)
                    #print("batch_rcv_amps_pred")
                    #print(np.shape(batch_rcv_amps_pred))
                    batch_rcv_amps_pred = batch_rcv_amps_pred-batch_rcv_amps_cte
                    batch_rcv_amps_pred_max, _ = torch.abs(batch_rcv_amps_pred).max(dim=0, keepdim=True)
                    # Normalize amplitudes by dividing by the maximum amplitude of each receiver
                    batch_rcv_amps_pred_norm = batch_rcv_amps_pred / (batch_rcv_amps_pred_max.abs() + 1e-10)
                    ##############batch_rcv_amps_pred_norm = batch_rcv_amps_pred
                    
                    #print("shape of receiver amplitudes predicted")
                    # print(np.shape(batch_rcv_amps_pred))
                    lossinner = criterion(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
                    
                    #########model2.grad[0:26,:] = 0
                    #filen = './deepwave/epoch1'+str(epoch)+'.npy'
                    #np.save(filen,net1out1.cpu().detach().numpy())
                    ##############if (epoch == num_epochs-1):
                    ##########    sumlossinner += lossinner.item()
                    #########if (epoch1 > lstart):
                    lossinner.backward()
                    net1out1.grad[(true[0,0,:,:]==2000)] = 0
                    #po = (true[:,:,:,:] == 2000)
                    #po = 1 - po.float()
                    #po = tgm.image.gaussian_blur(po, (5, 5), (8.0, 8.0))
                    #print("type :", net1out1.type())
                    #net1out1.grad = net1out1.grad*po[0,0,:,:]
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
        #np.save('./marmousi/po.npy',po.cpu().detach().numpy())
        #np.save('./deepwave/seis231.npy',batch_rcv_amps_pred.cpu().detach().numpy())
        #net1out1 = (net1out1 - 2000)/(4500-2000)
        #net1out1 = (net1out1-2000)/(4500-2000)
        #net1out1.grad = net1out1.grad*1000
                 
        return net1out1.grad
    # def sample(self,
    #            num_samples:int,
    #            current_device: int, **kwargs) -> Tensor:
    #     """
    #     Samples from the latent space and return the corresponding
    #     image space map.
    #     :param num_samples: (Int) Number of samples
    #     :param current_device: (Int) Device to run the model
    #     :return: (Tensor)
    #     """
    #     z = torch.randn(num_samples,
    #                     self.latent_dim)

    #     z = z.to(current_device)

    #     samples = self.decode(z)
    #     return samples


class VaeNoPhy_Net(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(VaeNoPhy_Net, self).__init__()
        self.is_deconv = True
        self.in_channels = outer_nc
        self.is_batchnorm = True
        self.n_classes = inner_nc

        filters = [64, 128, 256, 512, 1024]
        latent_dim = 256

        self.down1 = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2 = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3 = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4 = unetDown(filters[2], filters[3], self.is_batchnorm)
        #self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.fc_mu = nn.Linear(filters[-2]*25*7, latent_dim)
        self.fc_var = nn.Linear(filters[-2]*25*7, latent_dim)
        

        self.decoder_input = nn.Linear(latent_dim, filters[-2]*25*7)

        self.up4 = autoUp(filters[3], filters[3], self.is_deconv)
        self.up3 = autoUp(filters[3], filters[2], self.is_deconv)
        self.up2 = autoUp(filters[2], filters[1], self.is_deconv)
        self.up1 = autoUp(filters[1], filters[0], self.is_deconv)
        self.f1 = nn.Conv2d(filters[0], self.n_classes, 1)
        self.final = nn.ReLU(inplace=True)

    def encode(self, inputs):
        label_dsp_dim = (101,101)
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        #center = self.center(down4)
        
        #print("shape of down")
        #print(np.shape(down4))

        result = torch.flatten(down4, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        #center = self.center(down4)

        #print("shape of down4")
        #print(np.shape(down4))
        return [mu, log_var]

    def decode(self, inputs):
        filters = [64, 128, 256, 512, 1024]
        label_dsp_dim = (101,101)
        decoder_input = self.decoder_input(inputs)
        decoder_input = decoder_input.view(-1, filters[-2], 25, 7)
        up4 = self.up4(decoder_input)
        up3 = self.up3(up4)
        up2 = self.up2(up3)
        up1 = self.up1(up2)
        up1 = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        f1  = self.f1(up1)
        return self.final(f1)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
        

    # def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
    #     """
    #     Reparameterization trick to sample from N(mu, var) from
    #     N(0,1).
    #     :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    #     :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    #     :return: (Tensor) [B x D]
    #     """
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return eps * std + mu

    def forward(self, inputs, lstart, epoch1):
        
        mu,log_var = self.encode(inputs[:,:,1:800:2,:])
        #mu = torch.randn(1,256).to(inputs.get_device())
        #log_var = torch.randn(1,256).to(inputs.get_device())
        z = self.reparameterize(mu, log_var)
        de1 = self.decode(z)  
        de2 = 0*de1
        return  de1, mu, log_var, de2

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
    
    
    # def sample(self,
    #            num_samples:int,
    #            current_device: int, **kwargs) -> Tensor:
    #     """
    #     Samples from the latent space and return the corresponding
    #     image space map.
    #     :param num_samples: (Int) Number of samples
    #     :param current_device: (Int) Device to run the model
    #     :return: (Tensor)
    #     """
    #     z = torch.randn(num_samples,
    #                     self.latent_dim)

    #     z = z.to(current_device)

    #     samples = self.decode(z)
    #     return samples
    
    
class VaeLatentNoPhy_Net(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(VaeLatentNoPhy_Net, self).__init__()
        self.is_deconv = True
        self.in_channels = outer_nc
        self.is_batchnorm = True
        self.n_classes = inner_nc

        filters = [64, 128, 256, 512, 1024]
        latent_dim = 64

        self.down1 = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2 = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3 = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4 = unetDown(filters[2], filters[3], self.is_batchnorm)
        #self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.fc_mu = nn.Linear(filters[-2]*25*7, latent_dim)
        self.fc_var = nn.Linear(filters[-2]*25*7, latent_dim)
        

        self.decoder_input = nn.Linear(latent_dim, filters[-2]*25*7)

        self.up4 = autoUp(filters[3], filters[3], self.is_deconv)
        self.up3 = autoUp(filters[3], filters[2], self.is_deconv)
        self.up2 = autoUp(filters[2], filters[1], self.is_deconv)
        self.up1 = autoUp(filters[1], filters[0], self.is_deconv)
        self.f1 = nn.Conv2d(filters[0], self.n_classes, 1)
        self.final = nn.ReLU(inplace=True)

    def encode(self, inputs):
        label_dsp_dim = (101,101)
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        #center = self.center(down4)
        
        #print("shape of down")
        #print(np.shape(down4))

        result = torch.flatten(down4, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        #center = self.center(down4)

        #print("shape of down4")
        #print(np.shape(down4))
        return [mu, log_var]

    def decode(self, inputs):
        filters = [64, 128, 256, 512, 1024]
        label_dsp_dim = (101,101)
        decoder_input = self.decoder_input(inputs)
        decoder_input = decoder_input.view(-1, filters[-2], 25, 7)
        up4 = self.up4(decoder_input)
        up3 = self.up3(up4)
        up2 = self.up2(up3)
        up1 = self.up1(up2)
        up1 = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        f1  = self.f1(up1)
        return self.final(f1)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
        

    # def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
    #     """
    #     Reparameterization trick to sample from N(mu, var) from
    #     N(0,1).
    #     :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    #     :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    #     :return: (Tensor) [B x D]
    #     """
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return eps * std + mu

    def forward(self, inputs1, inputs2, lstart, epoch1):
        #mu,log_var = self.encode(inputs[:,:,1:800:2,:])
        #mu = torch.randn(1,256).to(inputs.get_device())
        #log_var = torch.randn(1,256).to(inputs.get_device())
        #z = self.reparameterize(mu, log_var)
        de1 = self.decode(inputs1)  
        de2 = 0*de1
        if (epoch1 > lstart):            
            de2 = self.prop(inputs2, de1, lstart, epoch1)
            de2 = torch.unsqueeze(de2,0)
            de2 = torch.unsqueeze(de2,0)
            #print("shape of de2")
            #print(np.shape(de2)) 
        return  de1, de2

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
                    
    def prop(self, inputs, vel, lstart, epoch1):
            #---------deepwave------------#
        net1out1 = vel * 100
        #print("---shape of vel---", str(np.shape(vel)))
        net1out1 = net1out1.detach()
        net1out1 = torch.squeeze(net1out1)
        devicek = net1out1.get_device()
        
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
        source_amplitudes_true = source_amplitudes_true.to(devicek)
        #lstart = -1
        num_batches = 1
        num_epochs = 1
        if (epoch1 > lstart):
            num_epochs = 100
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

        receiver_amplitudes_true = inputs[0,:,:,:]/10
        receiver_amplitudes_true = receiver_amplitudes_true.swapaxes(0,1)
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
        mat2 = mat2 * 1500.0
        #mat2 = torch.clamp(mat2,min=1500,max=4400)
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
        rcv_amps_true_max, _ = torch.abs(receiver_amplitudes_true).max(dim=0, keepdim=True)
        rcv_amps_true_norm = receiver_amplitudes_true / (rcv_amps_true_max.abs() + 1e-10)

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
                    model2 = torch.clamp(net1out1,min=1500,max=4400)
                    #np.save('before108.npy',net1out1.cpu().detach().numpy())
                    #net1out1 = torch.clamp(net1out1,min=2000,max=4500)
                    prop = deepwave.scalar.Propagator({'vp': model2}, dx)
                    batch_src_amps = source_amplitudes_true.repeat(
                        1, num_shots_per_batch, 1)
                    #print("shape of batch src amps")
                    #print(np.shape(batch_src_amps))
                    ############batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches].to(self.devicek)
                    batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches]
                    batch_x_s = x_s[it::num_batches].to(devicek)
                    ##################batch_x_s = x_s[it::num_batches]
                    #print("shape of batch src amps")
                    # print(np.shape(batch_x_s))
                    #####################batch_x_r = x_r[it::num_batches]
                    batch_x_r = x_r[it::num_batches].to(devicek)
                    #print("shape of batch receiver amps")
                    # print(np.shape(batch_x_r))
                    batch_rcv_amps_pred = prop(
                        batch_src_amps, batch_x_s, batch_x_r, dt)
                    #print("batch_rcv_amps_pred")
                    #print(np.shape(batch_rcv_amps_pred))
                    batch_rcv_amps_cte = receiver_amplitudes_cte[:,it::num_batches]
                    batch_rcv_amps_pred = batch_rcv_amps_pred - batch_rcv_amps_cte
                    batch_rcv_amps_pred_max, _ = torch.abs(batch_rcv_amps_pred).max(dim=0, keepdim=True)
                    # Normalize amplitudes by dividing by the maximum amplitude of each receiver
                    batch_rcv_amps_pred_norm = batch_rcv_amps_pred / (batch_rcv_amps_pred_max.abs() + 1e-10)
                    ##############batch_rcv_amps_pred_norm = batch_rcv_amps_pred
                    
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
        return net1out1
    
    
class VaeLatent2NoPhy_Net(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(VaeLatent2NoPhy_Net, self).__init__()
        self.is_deconv = True
        self.in_channels = outer_nc
        self.is_batchnorm = True
        self.n_classes = inner_nc

        filters = [64, 128, 256, 512, 1024]
        latent_dim = 512

        self.down1 = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2 = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3 = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4 = unetDown(filters[2], filters[3], self.is_batchnorm)
        #self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.fc_mu = nn.Linear(filters[-2]*13*10, latent_dim)
        self.fc_var = nn.Linear(filters[-2]*13*10, latent_dim)
        

        self.decoder_input = nn.Linear(latent_dim, filters[-2]*13*10)

        self.up4 = autoUp(filters[3], filters[3], self.is_deconv)
        self.up3 = autoUp(filters[3], filters[2], self.is_deconv)
        self.up2 = autoUp(filters[2], filters[1], self.is_deconv)
        self.up1 = autoUp(filters[1], filters[0], self.is_deconv)
        self.f1 = nn.Conv2d(filters[0], self.n_classes, 1)
        self.final = nn.ReLU(inplace=True)

    def encode(self, inputs):
        label_dsp_dim = (151,201)
        #print(" shape of inputs")
        #print(np.shape(inputs))
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        #center = self.center(down4)
        
        #print("shape of down")
        #print(np.shape(down4))

        result = torch.flatten(down4, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        #center = self.center(down4)

        #print("shape of down4")
        #print(np.shape(down4))
        return [mu,log_var]

    def decode(self, inputs):
        filters = [64, 128, 256, 512, 1024]
        label_dsp_dim = (151,201)
        decoder_input = self.decoder_input(inputs)
        decoder_input = decoder_input.view(-1, filters[-2], 10, 13)
        #print("decoder input :", np.shape(decoder_input))
        up4 = self.up4(decoder_input)
        up3 = self.up3(up4)
        up2 = self.up2(up3)
        up1 = self.up1(up2)
        #print("shape of up1 before:", np.shape(up1))
        up1 = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        #print("shape of up1 after:", np.shape(up1))
        f1  = self.f1(up1)
        return self.final(f1)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
        

    # def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
    #     """
    #     Reparameterization trick to sample from N(mu, var) from
    #     N(0,1).
    #     :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    #     :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    #     :return: (Tensor) [B x D]
    #     """
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return eps * std + mu

    def forward(self, inputs1, inputs2, lstart, epoch1):
        [mu,log_var] = self.encode(inputs1)
        ###mu = torch.randn(1,64).to(inputs2.get_device())
        ######log_var = torch.randn(1,64).to(inputs2.get_device())S
        z = self.reparameterize(mu, log_var)
        de1 = self.decode(z)  
        #de1[:,:,0:26,:] = 15.0
        
        #print("decoded")
        #print(de1)
        ##mu = 0*de1 #####switch of for physics guided
        ##log_var = 0*de1 #####switch of for physics guided
        de2 = 0*de1
        ##z = 0*de1 #####switch of for physics guided
        if (epoch1 > lstart):            
            de2 = self.prop(inputs2, de1, lstart, epoch1)
            #de2 = torch.unsqueeze(de2,0)
            #de2 = torch.unsqueeze(de2,0)
            #print("shape of de2")
            #print(np.shape(de2)) 
        return  de1, mu, log_var, z, de2

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
                    
    def prop(self, inputs, vel, lstart, epoch1):
            #---------deepwave------------#
            
        torch.cuda.set_device(7)  #RB Necessary if device <> 0
        GPU_string='cuda:'+str(7)
        devicek = torch.device(GPU_string)
        
        net1out1 = vel * 100
        #print("---shape of vel---", str(np.shape(vel)))
        net1out1 = net1out1.detach()
        net1out1 = torch.squeeze(net1out1)
        #devicek = net1out1.get_device()
        net1out1 = net1out1.to(devicek)
        #######net1out1[0:26,:] = 1500.0
        
        freq = 14
        dx = 10
        nt = 4001
        dt = 0.001
        num_shots = 16
        num_receivers_per_shot = 201
        num_sources_per_shot = 1
        num_dims = 2
        #ModelDim = [201,301]
        source_spacing = 201 * dx / num_shots
        receiver_spacing = 201 * dx / num_receivers_per_shot
        x_s = torch.zeros(num_shots, num_sources_per_shot, num_dims)
        x_s[:, 0, 1] = torch.arange(num_shots).float() * source_spacing
        x_r = torch.zeros(num_shots, num_receivers_per_shot, num_dims)
        x_r[0, :, 1] = torch.arange(
            num_receivers_per_shot).float() * receiver_spacing
        x_r[:, :, 1] = x_r[0, :, 1].repeat(num_shots, 1)

        source_amplitudes_true = (deepwave.wavelets.ricker(freq, nt, dt, 1/freq)
                                  .reshape(-1, 1, 1))
        #print("device ordinal :", self.devicek)
        source_amplitudes_true = source_amplitudes_true.to(devicek)
        #lstart = -1
        num_batches = 4
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
        mat2 = mat2 * 1500.0
        #mat2 = torch.clamp(mat2,min=1500,max=4400)
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
        rcv_amps_true_max, _ = torch.abs(receiver_amplitudes_true).max(dim=0, keepdim=True)
        rcv_amps_true_norm = receiver_amplitudes_true / (rcv_amps_true_max.abs() + 1e-10)

        criterion = torch.nn.MSELoss()

        #print("shape of mat2 :", np.shape(mat2))
        
        #Shuffle shot coordinates
        idx = torch.randperm(num_shots)
        x_s = x_s.view(-1,2)[idx].view(x_s.size())
        #RB Shuffle true's seismograms sources with same random values
        rcv_amps_true_norm = rcv_amps_true_norm[:,idx,:]
        #RB Shuffle direct wave seismograms sources with the same random values
        receiver_amplitudes_cte = receiver_amplitudes_cte[:,idx,:]


        if (epoch1 > lstart):
            net1out1.requires_grad = True
            optimizer2 = torch.optim.Adam([{'params': [net1out1], 'lr':10}])

        for epoch in range(num_epochs):
                for it in range(num_batches):
                    if (epoch1 > lstart):
                        optimizer2.zero_grad()
                    model2 = net1out1.clone()
                    model2 = torch.clamp(net1out1,min=2400,max=5500)
                    #np.save('before108.npy',net1out1.cpu().detach().numpy())
                    #net1out1 = torch.clamp(net1out1,min=2000,max=4500)
                    prop = deepwave.scalar.Propagator({'vp': model2}, dx)
                    batch_src_amps = source_amplitudes_true.repeat(
                        1, num_shots_per_batch, 1)
                    #print("shape of batch src amps")
                    #print(np.shape(batch_src_amps))
                    ############batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches].to(self.devicek)
                    batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches]
                    batch_x_s = x_s[it::num_batches].to(devicek)
                    ##################batch_x_s = x_s[it::num_batches]
                    #print("shape of batch src amps")
                    # print(np.shape(batch_x_s))
                    #####################batch_x_r = x_r[it::num_batches]
                    batch_x_r = x_r[it::num_batches].to(devicek)
                    #print("shape of batch receiver amps")
                    # print(np.shape(batch_x_r))
                    batch_rcv_amps_pred = prop(
                        batch_src_amps, batch_x_s, batch_x_r, dt)
                    #print("batch_rcv_amps_pred")
                    #print(np.shape(batch_rcv_amps_pred))
                    batch_rcv_amps_cte = receiver_amplitudes_cte[:,it::num_batches]
                    #batch_rcv_amps_pred = batch_rcv_amps_pred - batch_rcv_amps_cte
                    batch_rcv_amps_pred = batch_rcv_amps_pred
                    batch_rcv_amps_pred_max, _ = torch.abs(batch_rcv_amps_pred).max(dim=0, keepdim=True)
                    # Normalize amplitudes by dividing by the maximum amplitude of each receiver
                    batch_rcv_amps_pred_norm = batch_rcv_amps_pred / (batch_rcv_amps_pred_max.abs() + 1e-10)
                    ##############batch_rcv_amps_pred_norm = batch_rcv_amps_pred
                    
                    #print("shape of receiver amplitudes predicted")
                    # print(np.shape(batch_rcv_amps_pred))
                    lossinner = criterion(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
                    
                    #########model2.grad[0:26,:] = 0
                    #filen = './deepwave/epoch1'+str(epoch)+'.npy'
                    #np.save(filen,net1out1.cpu().detach().numpy())
                    ##############if (epoch == num_epochs-1):
                    ##########    sumlossinner += lossinner.item()
                    #########if (epoch1 > lstart):
                    lossinner.backward()
                    ##########net1out1.grad[0:26,:] = 0
                    ##########optimizer2.step()
                    #epoch_loss += loss.item()
                    #optimizer2.step()
        #if (epoch1 == 52): 
        #np.save('./deepwave/after1.npy',net1out1.cpu().detach().numpy())
        #np.save('./deepwave/seis231.npy',batch_rcv_amps_pred.cpu().detach().numpy())
        #net1out1 = (net1out1 - 2000)/(4500-2000)
        #net1out1 = net1out1/100           
        return net1out1.grad
    
    
    # def sample(self,
    #            num_samples:int,
    #            current_device: int, **kwargs) -> Tensor:
    #     """
    #     Samples from the latent space and return the corresponding
    #     image space map.
    #     :param num_samples: (Int) Number of samples
    #     :param current_device: (Int) Device to run the model
    #     :return: (Tensor)
    #     """
    #     z = torch.randn(num_samples,
    #                     self.latent_dim)

    #     z = z.to(current_device)

    #     samples = self.decode(z)
    #     return samples
    

class Flow(transform.Transform, nn.Module):
    
    def __init__(self):
        transform.Transform.__init__(self)
        nn.Module.__init__(self)
    
    # Init all parameters
    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)
            
    # Hacky hash bypass
    def __hash__(self):
        return nn.Module.__hash__(self)
    

# Main class for normalizing flow
class NormalizingFlow(nn.Module):

    def __init__(self, dim, blocks, flow_length, density):
        super().__init__()
        biject = []
        self.n_params = []
        for f in range(flow_length):
            for b_flow in blocks:
                cur_block = b_flow(dim)
                biject.append(cur_block)
                self.n_params.append(cur_block.n_parameters())
        self.transforms = transform.ComposeTransform(biject)
        self.bijectors = nn.ModuleList(biject)
        self.base_density = density
        self.final_density = distrib.TransformedDistribution(density, self.transforms)
        self.log_det = []
        self.dim = dim

    def forward(self, z):
        self.log_det = []
        # Applies series of flows
        for b in range(len(self.bijectors)):
            self.log_det.append(self.bijectors[b].log_abs_det_jacobian(z))
            z = self.bijectors[b](z)
        return z, self.log_det
        
    def n_parameters(self):
        return sum(self.n_params)
    
    def set_parameters(self, params):
        param_list = params.split(self.n_params, dim = 1)
        # Applies series of flows
        for b in range(len(self.bijectors)):
            self.bijectors[b].set_parameters(param_list[b])
            


class PlanarFlow(Flow):

    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.weight = []
        self.scale = []
        self.bias = []
        self.dim = dim
        self.domain = torch.distributions.constraints.Constraint()
        self.codomain = torch.distributions.constraints.Constraint()

    def _call(self, z):
        z = z.unsqueeze(2)
        f_z = torch.bmm(self.weight, z) + self.bias
        return (z + self.scale * torch.tanh(f_z)).squeeze(2)

    def log_abs_det_jacobian(self, z):
        z = z.unsqueeze(2)
        #print("size of z :", str(z.size()))
        #print("size of weight :", str(self.weight.size()))
        f_z = torch.bmm(self.weight, z) + self.bias
        psi = self.weight * (1 - torch.tanh(f_z) ** 2)
        det_grad = 1 + torch.bmm(psi, self.scale)
        return torch.log(det_grad.abs() + 1e-9)
    
    def set_parameters(self, p_list):
        self.weight = p_list[:, :self.dim].unsqueeze(1)
        self.scale = p_list[:, self.dim:self.dim*2].unsqueeze(2)
        self.bias = p_list[:, self.dim*2].unsqueeze(1).unsqueeze(2)
        
    def n_parameters(self):
        return 2 * self.dim + 1
    
       
    
class VaeNormalizing_Net(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(VaeNormalizing_Net, self).__init__()
        self.is_deconv = True
        self.in_channels = outer_nc
        self.is_batchnorm = True
        self.n_classes = inner_nc

        filters = [16, 32, 64, 128, 512]
        latent_dim = 64
        


        self.down1 = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2 = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3 = unetDown(filters[1], filters[2], self.is_batchnorm)
        #self.down4 = unetDown(filters[2], filters[3], self.is_batchnorm)
        #self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.fc_mu = nn.Linear(filters[2]*50*13, latent_dim)
        self.fc_sigma = nn.Sequential(nn.Linear(filters[2]*50*13, latent_dim),nn.Softplus(),nn.Hardtanh(min_val=1e-4, max_val=5.))
        
        self.flow = NormalizingFlow(dim=64, blocks=[PlanarFlow], flow_length=16, density=distrib.MultivariateNormal(torch.zeros(2), torch.eye(2)))
        self.flow_enc = nn.Linear(filters[2]*50*13, self.flow.n_parameters())
        self.flow_enc.weight.data.uniform_(-0.01, 0.01)
        

        self.decoder_input = nn.Linear(latent_dim, filters[3]*50*13)

        #self.up4 = autoUp(filters[3], filters[3], self.is_deconv)
        self.up3 = autoUp(filters[3], filters[2], self.is_deconv)
        self.up2 = autoUp(filters[2], filters[1], self.is_deconv)
        self.up1 = autoUp(filters[1], filters[0], self.is_deconv)
        self.f1 = nn.Conv2d(filters[0], self.n_classes, 1)
        self.final = nn.ReLU(inplace=True)

    def encode(self, inputs):
        label_dsp_dim = (101,101)
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        #down4 = self.down4(down3)
        #center = self.center(down4)
        
        #print("shape of down")
        #print(np.shape(down4))

        result = torch.flatten(down3, start_dim=1)
        mu = self.fc_mu(result)
        sigma = self.fc_sigma(result)
        flow_params = self.flow_enc(result)
        #center = self.center(down4)

        #print("shape of down4")
        #print(np.shape(down4))
        return [mu, sigma, flow_params]

    def decode(self, inputs):
        filters = [16, 32, 64, 128, 512]
        label_dsp_dim = (101,101)
        decoder_input = self.decoder_input(inputs)
        
        decoder_input = decoder_input.view(-1, filters[3], 50, 13)
        #print("shape of decoder input :", np.shape(decoder_input))
        #up4 = self.up4(decoder_input)
        #print("shape of up4 :", np.shape(up4))
        up3 = self.up3(decoder_input)
        #print("shape of up3 :", np.shape(up3))
        up2 = self.up2(up3)
        #print("shape of up2 :", np.shape(up2))
        up1 = self.up1(up2)
        #print("shape of up1 :", np.shape(up1))
        up1 = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        f1  = self.f1(up1)
        return self.final(f1)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
        

    # def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
    #     """
    #     Reparameterization trick to sample from N(mu, var) from
    #     N(0,1).
    #     :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    #     :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    #     :return: (Tensor) [B x D]
    #     """
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return eps * std + mu

    def forward(self, inputs1, inputs2, lstart, epoch1, p, lowf):
        z_params = self.encode(inputs2[:,:,1:800:2,:])
        #mu,log_var = z_params
        #z = self.reparameterize(mu, log_var)
        z_tilde, kl_div = self.latent(z_params)
        #print("shape of z_tilde :", np.shape(z_tilde))
        de1 = self.decode(z_tilde)  
        de1 = 2000 + de1*(4500-2000)
        #print("shape of de1 :", np.shape(de1))
        grad = 0*de1
        if (epoch1 > lstart):
            grad = self.prop(inputs2, de1, lstart, epoch1, inputs1)
            grad = torch.unsqueeze(grad,0)
            grad = torch.unsqueeze(grad,0)
            #print("shape of de2")
            #print(np.shape(de2))   
        return  de1, kl_div, grad

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
                    
                    
    def latent(self, z_params):
            n_batch = 1
            # Split the encoded values to retrieve flow parameters
            mu, sigma, flow_params = z_params
            
            #sigma = torch.exp(0.5*log_var)
            # Re-parametrize a Normal distribution
            q = distrib.Normal(torch.zeros(mu.shape[1]), torch.ones(sigma.shape[1]))
            dev = mu.get_device()
            
            #q = q.to(dev)
            # Obtain our first set of latent points
            z_0 = (sigma * q.sample((n_batch, )).to(dev)) + mu
            # Update flows parameters
            self.flow.set_parameters(flow_params)
            # Complexify posterior with flows
            z_k, list_ladj = self.flow(z_0)
            
            #print("z_k :",z_k)
            # ln p(z_k) 
            log_p_zk = torch.sum(-0.5 * z_k * z_k, dim=1)
            # ln q(z_0)  (not averaged)
            log_q_z0 = torch.sum(-0.5 * (sigma.log() + (z_0 - mu) * (z_0 - mu) * sigma.reciprocal()), dim=1)
            #  ln q(z_0) - ln p(z_k)
            logs = (log_q_z0 - log_p_zk).sum()
            # Add log determinants
            ladj = torch.cat(list_ladj, dim=1)
            # ln q(z_0) - ln p(z_k) - sum[log det]
            logs -= torch.sum(ladj)
            return z_k, (logs / float(n_batch))
        
    # forward modeling to compute gradients
    def prop(self, inputs, vel, lstart, epoch1, true):
        
        #torch.cuda.set_device(7)  #RB Necessary if device <> 0
        #GPU_string='cuda:'+str(7)
        #devicek = torch.device(GPU_string)
        net1out1 = vel
        #net1out1 = vel
        #net1out1 = (3550-1500)*vel+1500
        #print("---shape of vel---", str(np.shape(vel)))
        net1out1 = net1out1.detach()
        net1out1 = torch.squeeze(net1out1)
        #net1out1 = net1out1.to(devicek)
        devicek = net1out1.get_device()
        #net1out1[0:26,:] = 1500.0

        
        freq = 30
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
        source_amplitudes_true = source_amplitudes_true.to(devicek)
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
        min1 = torch.min(net1out1)
        #print("min1 :", min1)
        #print(min1.get_device())
        #min1 = min1.to(self.device1)
        mat2 = torch.ones(net1out1.size()[0],net1out1.size()[1]).to(devicek)
        mat2 = mat2 * 2000.0
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
        
        receiver_amplitudes_true = receiver_amplitudes_true-receiver_amplitudes_cte
        
        #print("receiver_amplitudes_true :", np.shape(receiver_amplitudes_true))
        #print("receiver_amplitudes_cte :", np.shape(receiver_amplitudes_cte))
        #receiver_amplitudes_true = receiver_amplitudes_true
        rcv_amps_true_max, _ = torch.abs(receiver_amplitudes_true).max(dim=0, keepdim=True)
        rcv_amps_true_norm = receiver_amplitudes_true / (rcv_amps_true_max.abs() + 1e-10)

        criterion = torch.nn.MSELoss()

        #print("shape of mat2 :", np.shape(mat2))
        

        if (epoch1 > lstart):
            net1out1.requires_grad = True
            optimizer2 = torch.optim.Adam([{'params': [net1out1], 'lr':10}])

        for epoch in range(num_epochs):
                #Shuffle shot coordinates
                idx = torch.randperm(num_shots)
                #idx = idx.type(torch.LongTensor)
                x_s = x_s.view(-1,2)[idx].view(x_s.size())
                #RB Shuffle true's seismograms sources with same random values
                rcv_amps_true_norm = rcv_amps_true_norm[:,idx,:]
                #RB Shuffle direct wave seismograms sources with the same random values
                receiver_amplitudes_cte = receiver_amplitudes_cte[:,idx,:]
        
                for it in range(num_batches):
                    #if (epoch1 > lstart):
                    optimizer2.zero_grad()
                    model2 = net1out1.clone()
                    model2 = torch.clamp(net1out1,min=2000,max=4500)
                    #np.save('before108.npy',net1out1.cpu().detach().numpy())
                    #net1out1 = torch.clamp(net1out1,min=2000,max=4500)
                    prop = deepwave.scalar.Propagator({'vp': model2}, dx)
                    batch_src_amps = source_amplitudes_true.repeat(1, num_shots_per_batch, 1)
                    #print("shape of batch src amps")
                    #print(np.shape(batch_src_amps))
                    ############batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches].to(self.devicek)
                    batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches]
                    batch_rcv_amps_cte = receiver_amplitudes_cte[:,it::num_batches]
                    batch_x_s = x_s[it::num_batches].to(devicek)
                    ##################batch_x_s = x_s[it::num_batches]
                    #print("shape of batch src amps")
                    # print(np.shape(batch_x_s))
                    #####################batch_x_r = x_r[it::num_batches]
                    batch_x_r = x_r[it::num_batches].to(devicek)
                    #print("shape of batch receiver amps")
                    # print(np.shape(batch_x_r))
                    batch_rcv_amps_pred = prop(batch_src_amps, batch_x_s, batch_x_r, dt)
                    #print("batch_rcv_amps_pred")
                    #print(np.shape(batch_rcv_amps_pred))
                    batch_rcv_amps_pred = batch_rcv_amps_pred-batch_rcv_amps_cte
                    batch_rcv_amps_pred_max, _ = torch.abs(batch_rcv_amps_pred).max(dim=0, keepdim=True)
                    # Normalize amplitudes by dividing by the maximum amplitude of each receiver
                    batch_rcv_amps_pred_norm = batch_rcv_amps_pred / (batch_rcv_amps_pred_max.abs() + 1e-10)
                    ##############batch_rcv_amps_pred_norm = batch_rcv_amps_pred
                    
                    #print("shape of receiver amplitudes predicted")
                    # print(np.shape(batch_rcv_amps_pred))
                    lossinner = criterion(batch_rcv_amps_pred_norm, batch_rcv_amps_true)
                    
                    #########model2.grad[0:26,:] = 0
                    #filen = './deepwave/epoch1'+str(epoch)+'.npy'
                    #np.save(filen,net1out1.cpu().detach().numpy())
                    ##############if (epoch == num_epochs-1):
                    ##########    sumlossinner += lossinner.item()
                    #########if (epoch1 > lstart):
                    lossinner.backward()
                    net1out1.grad[(true[0,0,:,:]==2000)] = 0
                    #po = (true[:,:,:,:] == 2000)
                    #po = 1 - po.float()
                    #po = tgm.image.gaussian_blur(po, (5, 5), (8.0, 8.0))
                    #print("type :", net1out1.type())
                    #net1out1.grad = net1out1.grad*po[0,0,:,:]
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
        #np.save('./marmousi/po.npy',po.cpu().detach().numpy())
        #np.save('./deepwave/seis231.npy',batch_rcv_amps_pred.cpu().detach().numpy())
        #net1out1 = (net1out1 - 2000)/(4500-2000)
        #net1out1 = (net1out1-2000)/(4500-2000)
        #net1out1.grad = net1out1.grad*1000
                 
        return net1out1.grad
    


class VaeNormalizingPhy_Net(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(VaeNormalizingPhy_Net, self).__init__()
        self.is_deconv = True
        self.in_channels = outer_nc
        self.is_batchnorm = True
        self.n_classes = inner_nc

        filters = [64, 128, 256, 512, 1024]
        latent_dim = 64
        


        self.down1 = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2 = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3 = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4 = unetDown(filters[2], filters[3], self.is_batchnorm)
        #self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.fc_mu = nn.Linear(filters[-2]*25*7, latent_dim)
        self.fc_sigma = nn.Sequential(nn.Linear(filters[-2]*25*7, latent_dim),nn.Softplus(),nn.Hardtanh(min_val=1e-4, max_val=5.))
        
        self.flow = NormalizingFlow(dim=64, blocks=[PlanarFlow], flow_length=6, density=distrib.MultivariateNormal(torch.zeros(2), torch.eye(2)))
        self.flow_enc = nn.Linear(filters[-2]*25*7, self.flow.n_parameters())
        self.flow_enc.weight.data.uniform_(-0.01, 0.01)
        

        self.decoder_input = nn.Linear(latent_dim, filters[-2]*25*7)

        self.up4 = autoUp(filters[3], filters[3], self.is_deconv)
        self.up3 = autoUp(filters[3], filters[2], self.is_deconv)
        self.up2 = autoUp(filters[2], filters[1], self.is_deconv)
        self.up1 = autoUp(filters[1], filters[0], self.is_deconv)
        self.f1 = nn.Conv2d(filters[0], self.n_classes, 1)
        self.final = nn.ReLU(inplace=True)

    def encode(self, inputs):
        label_dsp_dim = (101,101)
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        #center = self.center(down4)
        
        #print("shape of down")
        #print(np.shape(down4))

        result = torch.flatten(down4, start_dim=1)
        mu = self.fc_mu(result)
        sigma = self.fc_sigma(result)
        flow_params = self.flow_enc(result)
        #center = self.center(down4)

        #print("shape of down4")
        #print(np.shape(down4))
        return [mu, sigma, flow_params]

    def decode(self, inputs):
        filters = [64, 128, 256, 512, 1024]
        label_dsp_dim = (101,101)
        decoder_input = self.decoder_input(inputs)
        
        decoder_input = decoder_input.view(-1, filters[-2], 25, 7)
        #print("shape of decoder input :", np.shape(decoder_input))
        up4 = self.up4(decoder_input)
        #print("shape of up4 :", np.shape(up4))
        up3 = self.up3(up4)
        #print("shape of up3 :", np.shape(up3))
        up2 = self.up2(up3)
        #print("shape of up2 :", np.shape(up2))
        up1 = self.up1(up2)
        #print("shape of up1 :", np.shape(up1))
        up1 = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        f1  = self.f1(up1)
        return self.final(f1)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
        

    # def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
    #     """
    #     Reparameterization trick to sample from N(mu, var) from
    #     N(0,1).
    #     :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    #     :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    #     :return: (Tensor) [B x D]
    #     """
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return eps * std + mu

    def forward(self, inputs, lstart, epoch1):
        z_params = self.encode(inputs[:,:,1:800:2,:])
        #mu,log_var = z_params
        #z = self.reparameterize(mu, log_var)
        z_tilde, kl_div = self.latent(z_params)
        #print("shape of z_tilde :", np.shape(z_tilde))
        de1 = self.decode(z_tilde)  
        #print("shape of de1 :", np.shape(de1))
        de2 = 0*de1
        if (epoch1 > lstart):
            de2 = self.prop(inputs, de1, lstart, epoch1)
            de2 = torch.unsqueeze(de2,0)
            de2 = torch.unsqueeze(de2,0)
            #print("shape of de2")
            #print(np.shape(de2))    
        return  de1, kl_div, de2

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
                    
                    
    def latent(self, z_params):
            n_batch = 1
            # Split the encoded values to retrieve flow parameters
            mu, sigma, flow_params = z_params
            
            #sigma = torch.exp(0.5*log_var)
            # Re-parametrize a Normal distribution
            q = distrib.Normal(torch.zeros(mu.shape[1]), torch.ones(sigma.shape[1]))
            dev = mu.get_device()
            
            #q = q.to(dev)
            # Obtain our first set of latent points
            z_0 = (sigma * q.sample((n_batch, )).to(dev)) + mu
            # Update flows parameters
            self.flow.set_parameters(flow_params)
            # Complexify posterior with flows
            z_k, list_ladj = self.flow(z_0)
            
            #print("z_k :",z_k)
            # ln p(z_k) 
            log_p_zk = torch.sum(-0.5 * z_k * z_k, dim=1)
            # ln q(z_0)  (not averaged)
            log_q_z0 = torch.sum(-0.5 * (sigma.log() + (z_0 - mu) * (z_0 - mu) * sigma.reciprocal()), dim=1)
            #  ln q(z_0) - ln p(z_k)
            logs = (log_q_z0 - log_p_zk).sum()
            # Add log determinants
            ladj = torch.cat(list_ladj, dim=1)
            # ln q(z_0) - ln p(z_k) - sum[log det]
            logs -= torch.sum(ladj)
            return z_k, (logs / float(n_batch))
        
        
    def prop(self, inputs, vel, lstart, epoch1):
        
            #---------deepwave------------#
        net1out1 = vel * 100
        #print("---shape of vel---", str(np.shape(vel)))
        net1out1 = net1out1.detach()
        net1out1 = torch.squeeze(net1out1)
        devicek = net1out1.get_device()
        
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
        source_amplitudes_true = source_amplitudes_true.to(devicek)
        #lstart = -1
        num_batches = 1
        num_epochs = 1
        if (epoch1 > lstart):
            num_epochs = 10
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

        receiver_amplitudes_true = inputs[0,:,:,:]/10
        receiver_amplitudes_true = receiver_amplitudes_true.swapaxes(0,1)
        #print("shape of receiver amplitudes true")
        #print(np.shape(receiver_amplitudes_true))

        ######rcv_amps_true_norm = receiver_amplitudes_true
        #print("receiver amplitude true shape")
        # print(np.shape(receiver_amplitudes_true))
        #net1out1 = net1out.detach()
        #net1out1 = torch.tensor(net1out1)
        #net1out1 = net1out1*(4500-2000)+2000
        #print(np.shape(net1out1))
        min1 = torch.min(net1out1)
        #print(min1.get_device())
        #min1 = min1.to(self.device1)
        mat2 = torch.ones(net1out1.size()[0],net1out1.size()[1]).to(devicek)
        mat2 = mat2 * min1
        mat2 = torch.clamp(mat2,min=2000,max=4500)
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
        rcv_amps_true_max, _ = torch.abs(receiver_amplitudes_true).max(dim=0, keepdim=True)
        rcv_amps_true_norm = receiver_amplitudes_true / (rcv_amps_true_max.abs() + 1e-10)

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
                    #print(np.shape(batch_src_amps))
                    ############batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches].to(self.devicek)
                    batch_rcv_amps_true = rcv_amps_true_norm[:,it::num_batches]
                    batch_x_s = x_s[it::num_batches].to(devicek)
                    ##################batch_x_s = x_s[it::num_batches]
                    #print("shape of batch src amps")
                    # print(np.shape(batch_x_s))
                    #####################batch_x_r = x_r[it::num_batches]
                    batch_x_r = x_r[it::num_batches].to(devicek)
                    #print("shape of batch receiver amps")
                    # print(np.shape(batch_x_r))
                    batch_rcv_amps_pred = prop(
                        batch_src_amps, batch_x_s, batch_x_r, dt)
                    #print("batch_rcv_amps_pred")
                    #print(np.shape(batch_rcv_amps_pred))
                    batch_rcv_amps_cte = receiver_amplitudes_cte[:,it::num_batches]
                    batch_rcv_amps_pred = batch_rcv_amps_pred - batch_rcv_amps_cte
                    batch_rcv_amps_pred_max, _ = torch.abs(batch_rcv_amps_pred).max(dim=0, keepdim=True)
                    # Normalize amplitudes by dividing by the maximum amplitude of each receiver
                    batch_rcv_amps_pred_norm = batch_rcv_amps_pred / (batch_rcv_amps_pred_max.abs() + 1e-10)
                    ##############batch_rcv_amps_pred_norm = batch_rcv_amps_pred
                    
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
        return net1out1
    

class Vaevel_Net(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Vaevel_Net, self).__init__()
        self.is_deconv = True
        self.in_channels = outer_nc
        self.is_batchnorm = True
        self.n_classes = inner_nc

        filters = [64, 128, 256, 512, 1024]
        latent_dim = 128

        self.down1 = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2 = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3 = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4 = unetDown(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[3], self.is_batchnorm)

        self.fc_mu = nn.Linear(filters[-2]*13*19, latent_dim)
        self.fc_var = nn.Linear(filters[-2]*13*19, latent_dim)
        

        self.decoder_input = nn.Linear(latent_dim, filters[-2]*13*19)

        self.up4 = autoUp(filters[3], filters[3], self.is_deconv)
        self.up3 = autoUp(filters[3], filters[2], self.is_deconv)
        self.up2 = autoUp(filters[2], filters[1], self.is_deconv)
        self.up1 = autoUp(filters[1], filters[0], self.is_deconv)
        self.f1 = nn.Conv2d(filters[0], self.n_classes, 1)
        self.final = nn.ReLU(inplace=True)

    def encode(self, inputs):
        label_dsp_dim = (201,301)
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        #center = self.center(down4)
        

        result = torch.flatten(down4, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        #center = self.center(down4)

        #print("shape of down4")
        #print(np.shape(down4))
        return [mu, log_var]

    def decode(self, inputs):
        filters = [64, 128, 256, 512, 1024]
        label_dsp_dim = (201,301)
        decoder_input = self.decoder_input(inputs)
        decoder_input = decoder_input.view(-1, filters[-2], 13, 19)
        up4 = self.up4(decoder_input)
        up3 = self.up3(up4)
        up2 = self.up2(up3)
        up1 = self.up1(up2)
        up1 = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        f1  = self.f1(up1)
        return self.final(f1)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
        

    # def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
    #     """
    #     Reparameterization trick to sample from N(mu, var) from
    #     N(0,1).
    #     :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    #     :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    #     :return: (Tensor) [B x D]
    #     """
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return eps * std + mu

    def forward(self, inputs, lstart, epoch1):
        mu,log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        de1 = self.decode(z)  
        de2 = 0*de1
        return  de1, mu, log_var, de2

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
    
    
    # def sample(self,
    #            num_samples:int,
    #            current_device: int, **kwargs) -> Tensor:
    #     """
    #     Samples from the latent space and return the corresponding
    #     image space map.
    #     :param num_samples: (Int) Number of samples
    #     :param current_device: (Int) Device to run the model
    #     :return: (Tensor)
    #     """
    #     z = torch.randn(num_samples,
    #                     self.latent_dim)

    #     z = z.to(current_device)

    #     samples = self.decode(z)
    #     return samples
    