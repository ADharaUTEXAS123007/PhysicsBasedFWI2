"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from typing import OrderedDict
from options.train_options import TrainOptions
from data import create_dataset
from data import create_dataset2
from models import create_model
from util.visualizer import Visualizer
import numpy as np
#import ray

if __name__ == '__main__':
    #ray.init()
    print('run till here 0')
    opt = TrainOptions().parse()   # get training options
    print('run till here 1')
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset2 = create_dataset2(opt) #create dataset for validation
    dataset_size = len(dataset)    # get the number of images in the dataset.
    dataset2_size = len(dataset2) #get the number of images in validation dataset
    print('The number of training images = %d' % dataset_size)
    print('The number of validation images = %d' % dataset2_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    losses1 = OrderedDict()
    lstart = 0
    Lhist = np.ones(4)
    freqL = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    mop = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
         epoch_start_time = time.time()  # timer for entire epoch
         iter_data_time = time.time()    # timer for data loading per iteration
         epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
         visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

         model.eval()  #For going to validation 
         Validationloss = 0.0
         for k, data2 in enumerate(dataset2):
             model.set_input(data2)
             model.test()
             model.compute_loss_only()
             Validationloss = Validationloss + model.loss_V_MSE.item()

         #model.update_epoch(epoch)
         model.train()
         model.update_learning_rate()    # update learning rates in the beginning of every epoch.
         Modelloss = 0.0
         Dataloss = 0.0
         Model1loss = 0.0
         KLloss = 0.0
         for i, data in enumerate(dataset):  # inner loop within one epoch
             ##print("i: " + str(i))
             iter_start_time = time.time()  # timer for computation per iteration
             if total_iters % opt.print_freq == 0:
                 t_data = iter_start_time - iter_data_time

             total_iters += opt.batch_size
             epoch_iter += opt.batch_size
             model.set_input(data)         # unpack data from dataset and apply preprocessing
             model.optimize_parameters(epoch,i,lstart,freqL[mop])   # calculate loss functions, get gradients, update network weights
             #model.test()
             #if (i==190):
             #   visuals = model.get_current_visuals()
             #   print(visuals['real_B'])
             #print("model losses out of loop")
             #print(model.loss_M_MSE.item())
            #  if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            #     losses = model.get_current_losses()
            #     t_comp = (time.time() - iter_start_time) / opt.batch_size
            #     visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            #     print("---epoch----")
            #     print(epoch)
            #     print("--losses---")
            #     print(losses)
            #     if opt.display_id > 0:
            #         visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

             #if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
             #    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
             #    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
             #    model.save_networks(save_suffix)

             
             #if (i==259):
             #    print(data['A'])
             #    print(data['C'])
             #    model.print_values()
                 #print(model.fake_B)
                 #np.save('./datasets/testO/A.npy',data['A'].numpy())
                 #np.save('./datasets/testO/B.npy',data['B'].numpy())

             iter_data_time = time.time()
             Modelloss = Modelloss + model.loss_M_MSE.item()
             Dataloss = Dataloss + model.loss_D_MSE
             
             if (epoch < 4):
                 Lhist[epoch-1] = model.loss_D_MSE
             else:
                 Lhist[1] = Lhist[2]
                 Lhist[2] = Lhist[3]
                 Lhist[3] = model.loss_D_MSE
                 
             if (epoch > 4):
                 if (np.abs((Lhist[3]-Lhist[1])/Lhist[1]) <= .0009):
                     mop = mop + 1
                 
             #if (epoch > lstart):
             #   Model1loss = Model1loss + model.loss_M1_MSE.item()     
             #else:
             #Model1loss = Model1loss + model.loss_M1_MSE
                 
                
             KLloss = KLloss + model.loss_K_MSE
             #KLloss = KLloss + model.loss_K_MSE


         
         if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
             print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
             model.save_networks('latest')
             model.save_networks(epoch)

         if epoch % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
             save_result = total_iters % opt.update_html_freq == 0
             model.compute_visuals()
             visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        
         if epoch % opt.display_freq == 0:    #plot losses
            losses1['Modelloss'] = Modelloss/(i+1)
            losses1['Dataloss'] = Dataloss/(i+1)
            losses1['Validationloss'] = Validationloss/(k+1)
            #losses1['Model1loss'] = Model1loss/(i+1)
            losses1['KL divergence'] = KLloss/(i+1)
            #print(losses1)
            #losses2 = model.get_current_losses()
            #print(losses2)
            visualizer.plot_current_losses(epoch, 0, losses1)
            visualizer.print_current_losses(epoch, epoch_iter, losses1)
            
         print("Lhist :", Lhist)

         print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
