
no_proxy=localhost python trainVal4dVel.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/ --name velModel3 --model Vae --direction AtoB --input_nc 15 --output_nc 1 --display_port 9997 --n_epochs 200 --n_epochs_decay 100  --batch_size 25  --gpu_ids 0,1,2,3 --no_html --display_freq 1 --print_freq 200 --lr 0.0001 --verbose --save_epoch_freq 2
