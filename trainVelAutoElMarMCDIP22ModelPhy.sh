rm -rf ./marmousiEl17Apr
mkdir ./marmousiEl17Apr
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiElMarMCDIP22/*.txt
no_proxy=localhost python trainValLatent4dVel2Elastic.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiEl/ --name MarmousiElMarMCDIP22 --model AutoElMarMCDIP22 --direction AtoB --input_nc 35 --output_nc 1 --display_port 9998 --n_epochs 1500 --n_epochs_decay 500 --batch_size 8 --gpu_ids 1 --no_html --display_freq 1 --print_freq 1 --lr 0.005 --verbose --save_epoch_freq 10 
