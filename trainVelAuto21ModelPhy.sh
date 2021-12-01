
mkdir ./marmousi21
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiOpenFWI21/*.txt
no_proxy=localhost python trainValLatent4dVel2.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiF/ --name MarmousiOpenFWI21 --model Auto21 --direction AtoB --input_nc 18 --output_nc 1 --display_port 9996 --n_epochs 2000 --n_epochs_decay 2000  --batch_size 8 --gpu_ids 7 --no_html --display_freq 1 --print_freq 1 --lr 0.01 --verbose --save_epoch_freq 25
