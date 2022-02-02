rm -rf ./marmousiEl
mkdir ./marmousiEl
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl/*.txt
no_proxy=localhost python trainValLatent4dVel2Elastic.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiEl/ --name MarmousiEl22 --model AutoEl22 --direction AtoB --input_nc 28 --output_nc 1 --display_port 8889 --n_epochs 35 --n_epochs_decay 0 --batch_size 8 --gpu_ids 5 --no_html --display_freq 1 --print_freq 1 --lr 1 --verbose --save_epoch_freq 50 --init_type kaiming
