
no_proxy=localhost python trainValLatent4dVel.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/Fault/Phy --name VaePhysics --model VaeLatentNoPhy --direction AtoB --input_nc 10 --output_nc 1 --display_port 9997 --n_epochs 20000 --n_epochs_decay 0  --batch_size 1  --gpu_ids 2 --no_html --display_freq 1 --print_freq 15 --lr 0.0001 --verbose --save_epoch_freq 100 --epoch 950 --continue_train
