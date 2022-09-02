
mkdir ./marmousi21
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/SEAMOpenFWI21/*.txt
no_proxy=localhost python trainValLatent4dVel2.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/SEAMN/ --name SEAMOpenFWI21 --model Auto21 --direction AtoB --input_nc 20 --output_nc 1 --display_port 9996 --n_epochs 2500 --n_epochs_decay 3000  --batch_size 8 --gpu_ids 5 --no_html --display_freq 1 --print_freq 1 --lr 0.01 --verbose --save_epoch_freq 25
