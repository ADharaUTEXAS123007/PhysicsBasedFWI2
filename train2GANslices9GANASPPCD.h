
no_proxy=localhost python trainVal4d.py --dataroot /glb/data/eptr_am_2/Arnab/volumes/training_data/events/HorizonWindow/att_volumes/slices9/ --name pix2pix2slices9ASPPCD --model pix2pix2ASPP --direction AtoB --input_nc 2 --output_nc 1 --display_port 9997 --n_epochs 1400 --n_epochs_decay 1400 --batch_size 128  --gpu_ids 0,1 --no_html --display_freq 40 --print_freq 40 --verbose
