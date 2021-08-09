
no_proxy=localhost python trainVal4d.py --dataroot /glb/data/eptr_am_2/Arnab/volumes/training_data/events/HorizonWindow/att_volumes/slices11/ --name pix2pix2slices11ASPP --model pix2pix2ASPP --direction AtoB --input_nc 2 --output_nc 2 --display_port 9997 --n_epochs 1500 --n_epochs_decay 1500 --batch_size 128  --gpu_ids 0,1 --no_html --display_freq 40 --print_freq 40 --verbose --continue_train
