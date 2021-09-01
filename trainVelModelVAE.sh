
no_proxy=localhost python trainVal4dVel2.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/ --name VaeBased --model Vae --direction AtoB --input_nc 15 --output_nc 1 --display_port 9987 --n_epochs 100 --n_epochs_decay 50  --batch_size 7  --gpu_ids 0 --no_html --display_freq 1 --print_freq 200 --lr 0.0001 --verbose --save_epoch_freq 100
