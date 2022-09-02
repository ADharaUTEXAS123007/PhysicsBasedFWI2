
no_proxy=localhost python train4d.py --dataroot /glb/data/eptr_am_2/Arnab/volumes/training_data/events/HorizonWindow/att_volumes/slices/ --name unetHorizonSeismic --model unet --direction AtoB --input_nc 2 --output_nc 1 --display_port 9997 --n_epochs 500 --n_epochs_decay 500 --batch_size 64 --gpu_ids 0,1 --print_freq 50 --no_html --continue_train


