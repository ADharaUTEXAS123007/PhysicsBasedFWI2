
no_proxy=localhost python trainVal4d.py --dataroot /glb/data/eptr_am_2/Arnab/volumes/training_data/events/HorizonWindow/att_volumes/patches1 --name unetSeismicAB --model unet --direction AtoB --input_nc 2 --output_nc 1 --display_port 9997 --n_epochs 500 --n_epochs_decay 500 --batch_size 256 --gpu_ids 0,1 --no_html --display_freq 50 --verbose
