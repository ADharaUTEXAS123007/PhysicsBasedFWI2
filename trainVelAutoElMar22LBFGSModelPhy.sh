rm -rf ./marmousiEl9Mar
mkdir ./marmousiEl9Mar
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiElMar22/*.txt
no_proxy=localhost python trainValLatent4dVel2Elastic.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiEl/ --name MarmousiElMar22LBFGS --model AutoElMar22LBFGS --direction AtoB --input_nc 35 --output_nc 1 --display_port 9998 --n_epochs 1 --n_epochs_decay 0 --batch_size 8 --gpu_ids 1 --no_html --display_freq 1 --print_freq 1 --lr 0.005 --verbose --save_epoch_freq 10 --epoch 9 --continue_train
