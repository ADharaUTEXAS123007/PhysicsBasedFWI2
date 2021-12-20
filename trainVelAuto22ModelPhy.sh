
mkdir ./marmousi22
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiOpenFWI22/*.txt
no_proxy=localhost python trainValLatent4dVel2.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiN/ --name MarmousiOpenFWI22 --model Auto22 --direction AtoB --input_nc 18 --output_nc 1 --display_port 9990 --n_epochs 2500 --n_epochs_decay 2500 --batch_size 8 --gpu_ids 7 --no_html --display_freq 1 --print_freq 1 --lr 0.01 --verbose --save_epoch_freq 25
