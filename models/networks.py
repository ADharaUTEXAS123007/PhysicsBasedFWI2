import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import sys
import torch.nn.functional as F

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
from unet_layers import unetConv2

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
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, [gpu_ids])  # multi-GPUs
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
        net = Auto_Net(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout) 
    elif netG == 'Vae':
        net = Vae_Net(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout) 
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


#######################UNET 2##############################################
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()
        # Kernel size: 3*3, Stride: 1, Padding: 1
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.Dropout2D(0.8),
                                       nn.BatchNorm2d(out_size),
                                       nn.LeakyReLU(0.1,inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.LeakyReLU(0.1,inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True),)
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm)
        self.down = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, inputs):
        outputs = self.conv(inputs)
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
        self.is_batchnorm  = True
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
        self.final   = nn.ReLU(inplace=True)
        
    def forward(self, inputs):
        label_dsp_dim = (201,301)
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
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs2):
        outputs2 = self.up(inputs2)
        #offset1 = (outputs2.size()[2]-inputs1.size()[2])
        #offset2 = (outputs2.size()[3]-inputs1.size()[3])
        #padding=[offset2//2,(offset2+1)//2,offset1//2,(offset1+1)//2]
        # Skip and concatenate 
        #outputs1 = F.pad(inputs1, padding)
        return self.conv2(outputs2)

class Auto_Net(nn.Module):
    def __init__(self,outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Auto_Net, self).__init__()
        self.is_deconv     = True
        self.in_channels   = outer_nc
        self.is_batchnorm  = True
        self.n_classes     = inner_nc
        
        filters = [64, 128, 256, 512, 1024]
        
        self.down1   = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2   = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3   = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4   = unetDown(filters[2], filters[3], self.is_batchnorm)
        self.center  = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.up4     = autoUp(filters[4], filters[3], self.is_deconv)
        self.up3     = autoUp(filters[3], filters[2], self.is_deconv)
        self.up2     = autoUp(filters[2], filters[1], self.is_deconv)
        self.up1     = autoUp(filters[1], filters[0], self.is_deconv)
        self.f1      = nn.Conv2d(filters[0],self.n_classes, 1)
        self.final   = nn.ReLU(inplace=True)
        
    def forward(self, inputs):
        label_dsp_dim = (201,301)
        down1  = self.down1(inputs)
        down2  = self.down2(down1)
        down3  = self.down3(down2)
        down4  = self.down4(down3)
        center = self.center(down4)
        up4    = self.up4(center)
        up3    = self.up3(up4)
        up2    = self.up2(up3)
        up1    = self.up1(up2)
        up1    = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        f1     = self.f1(up1)

        #result = torch.flatten(f1, start_dim=1)
        print(" shape of result :", np.shape(down4))

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


class Vae_Net(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Vae_Net, self).__init__()
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
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.fc_mu = nn.Linear(filters[-1]*25*19, latent_dim)
        self.fc_var = nn.Linear(filters[-1]*25*19, latent_dim)
        

        self.decoder_input = nn.Linear(latent_dim, filters[-1]*25*19)

        self.up4 = autoUp(filters[4], filters[3], self.is_deconv)
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
        center = self.center(down4)

        result = torch.flatten(center, start_dim=1)
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
        decoder_input = decoder_input.view(-1, filters[-1], 25, 19)
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

    def forward(self, inputs):
        mu,log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        de1 = self.decode(z)
        return  de1, mu, log_var

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
