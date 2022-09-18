"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd.variable import Variable
import deepwave
import numpy as np

def eval_loss(net, criterion, loader, use_cuda=False):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    correct = 0
    total_loss = 0
    total = 0 # number of samples
    num_batch = len(loader)

    if use_cuda:
        net.cuda()
    net.eval()

    with torch.no_grad():
        if isinstance(criterion, nn.CrossEntropyLoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)
                targets = Variable(targets)
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets).sum().item()

        elif isinstance(criterion, nn.MSELoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)

                one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
                one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
                one_hot_targets = one_hot_targets.float()
                one_hot_targets = Variable(one_hot_targets)
                if use_cuda:
                    inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
                outputs = F.softmax(net(inputs))
                loss = criterion(outputs, one_hot_targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.cpu().eq(targets).sum().item()

    return total_loss/total, 100.*correct/total


def eval_loss2(net, use_cuda=False):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    total_loss = 0
    total = 1
    A_img = np.load('/disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/Simple/trainA/1.npy')
    A_img = np.expand_dims(A_img,0)
    A = torch.from_numpy(A_img)
    A = A.float()
    
    B_img = np.load('/disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/Simple/trainB/1.npy')
    B_img = np.expand_dims(B_img,0)
    B_img = np.expand_dims(B_img,0)
    B = torch.from_numpy(B_img)
    B = B.float()
    
    C_img = np.load('/disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/Simple/trainC/1.npy')
    C_img = np.expand_dims(C_img,0)
    C_img = np.expand_dims(C_img,0)
    C = torch.from_numpy(C_img)
    C = C.float()
    
    print("shape of A :", np.shape(A))
    latent = torch.ones(1,1,1,1)
    lstart = 1
    epoch1 = 2
    [fake_B,grad,latent,loss_D_MSE,down3,up2,up1] = net(B,A,lstart,epoch1,latent,C)
    
    print("loss D MSE :", loss_D_MSE)
    
    return loss_D_MSE

def eval_loss3(net, use_cuda=False):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    total_loss = 0
    total = 1
    A_img = np.load('/disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiEl/trainA/1.npy')
    A_img = np.expand_dims(A_img,0)
    A = torch.from_numpy(A_img)
    A = A.float()
    
    B_img = np.load('/disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiEl/trainB/1.npy')
    B_img = np.expand_dims(B_img,0)
    B_img = np.expand_dims(B_img,0)
    B = torch.from_numpy(B_img)
    B = B.float()
    
    C_img = np.load('/disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiEl/trainC/1.npy')
    C_img = np.expand_dims(C_img,0)
    C_img = np.expand_dims(C_img,0)
    C = torch.from_numpy(C_img)
    C = C.float()

    D_img = np.load('/disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiEl/trainD/1.npy')
    D_img = np.expand_dims(D_img,0)
    D_img = np.expand_dims(D_img,0)
    D = torch.from_numpy(D_img)
    D = D.float()
    
    print("shape of A :", np.shape(A))
    latent = torch.ones(1,1,1,1)
    lstart = 1
    epoch1 = 2
    freq = 20
    #[fake_B,grad,latent,loss_D_MSE,down3,up2,up1] = net(B,A,lstart,epoch1,latent,C)
    print("shape of A :", np.shape(A))
    print("shape of B :", np.shape(B))
    print("shape of C :", np.shspe(C))
    print("shape of D :", np.shape(D))


    [fake_Vp,fake_Vs,fake_Rho, grad,latent,vp_grad,vs_grad,rho_grad,loss_D_MSE] = net(B,A,lstart,epoch1,latent,C,D,freq)  # G(A)
    
    print("loss D MSE :", loss_D_MSE)
    
    return loss_D_MSE
