
no_proxy=localhost python trainVal4d.py --dataroot /glb/data/eptr_am_2/Arnab/volumes/training_data/events/HorizonWindow/att_volumes/patches3/ --name pix2pix2SeismicDRMSPatch3ResUNET --model pix2pix2ResUNET --direction AtoB --input_nc 3 --output_nc 1 --display_port 9997 --n_epochs 1400 --n_epochs_decay 1400 --batch_size 32  --gpu_ids 0,1 --no_html --display_freq 20 --verbose
