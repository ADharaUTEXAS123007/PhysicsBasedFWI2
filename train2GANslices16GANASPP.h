
no_proxy=localhost python trainVal4d.py --dataroot /glb/data/eptr_am_2/Arnab/volumes/training_data/events/HorizonWindow/att_volumes/slices20/ --name pix2pix2slices20ASPP --model pix2pix2ASPP --direction AtoB --input_nc 1 --output_nc 1 --display_port 9997 --n_epochs 500  --n_epochs_decay 500 --batch_size 72  --gpu_ids 0,1 --no_html --display_freq 40 --print_freq 40 --verbose --save_epoch_freq 100
