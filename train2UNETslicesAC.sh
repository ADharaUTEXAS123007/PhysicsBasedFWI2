
no_proxy=localhost python trainVal4d.py --dataroot /glb/data/eptr_am_2/Arnab/volumes/training_data/events/HorizonWindow/att_volumes/slices/ --name unetSeismicSliceAC --model unetAC --direction AtoC --input_nc 2 --output_nc 2 --display_port 9997 --n_epochs 500 --n_epochs_decay 500 --batch_size 64 --gpu_ids 0,1 --no_html --display_freq 10 --verbose
