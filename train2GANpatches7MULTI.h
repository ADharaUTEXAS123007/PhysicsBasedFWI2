
no_proxy=localhost python trainVal4d.py --dataroot /glb/data/eptr_am_2/Arnab/volumes/training_data/events/HorizonWindow/att_volumes/patches1/ --name pix2pix2SeismicDRMSPATCH1MULTI --model pix2pix2Multi --direction AtoB --input_nc 3 --output_nc 1 --display_port 9997 --n_epochs 1300 --n_epochs_decay 1300 --batch_size 128 --gpu_ids 0,1 --no_html --display_freq 5 --verbose
