
no_proxy=localhost python trainVal4d.py --dataroot /glb/data/eptr_am_2/Arnab/FCNVMB-Deep-learning-based-seismic-velocity-model-building/ --name velModel --model NewU --direction AtoB --input_nc 29 --output_nc 1 --display_port 9997 --n_epochs 100 --n_epochs_decay 50 --batch_size 10  --gpu_ids 0,1 --no_html --display_freq 1 --print_freq 1600 --lr 0.0001 --verbose
