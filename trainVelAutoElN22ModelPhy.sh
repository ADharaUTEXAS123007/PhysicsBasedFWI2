rm -rf ./marmousiEl12Apr
mkdir ./marmousiEl12Apr
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl22/*.txt
no_proxy=localhost python trainValLatent4dVel2ElasticModel.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiEl/ --name MarmousiEl22N --model AutoEl22N --direction AtoB --input_nc 28 --output_nc 1 --display_port 9998 --n_epochs 2000 --n_epochs_decay 500 --batch_size 8 --gpu_ids 1 --no_html --display_freq 1 --print_freq 1 --lr 0.005 --verbose --save_epoch_freq 5000
