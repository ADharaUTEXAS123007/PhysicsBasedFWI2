
no_proxy=localhost python trainVal4d.py --dataroot /glb/data/eptr_am_2/Arnab/volumes/training_data/events/HorizonWindow/att_volumes/slices4/ --name pix2pix2SeismicDRMSSlice4ASPP --model pix2pix2ASPP33 --direction AtoB --input_nc 3 --output_nc 3 --display_port 9997 --n_epochs 1000 --n_epochs_decay 1000 --batch_size 128 --gpu_ids 0,1 --no_html --display_freq 10 --verbose
