
no_proxy=localhost python trainValLatent4dVel2.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/Marmousi/ --name VaeLatent2NoPhy --model VaeLatent2NoPhy --direction AtoB --input_nc 1 --output_nc 1 --display_port 9996 --n_epochs 2000 --n_epochs_decay 0  --batch_size 1  --gpu_ids 7 --no_html --display_freq 2 --print_freq 1 --lr 0.0001 --verbose --save_epoch_freq 100
