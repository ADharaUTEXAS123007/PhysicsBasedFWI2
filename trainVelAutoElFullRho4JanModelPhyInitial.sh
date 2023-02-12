rm -rf ./marmousiEl4JanInit
mkdir ./marmousiEl4JanInit
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiElFull4JanInit/*.txt
no_proxy=localhost python trainValLatent4dVel2Elastic.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiElInit/ --name MarmousiElFull4JanInit --model AutoElFullRhoMar22 --direction AtoB --input_nc 46 --output_nc 1 --display_port 9998 --n_epochs 4000  --n_epochs_decay 2500 --batch_size 8 --gpu_ids 3 --no_html --display_freq 1 --print_freq 1 --lr 0.0025 --verbose --save_epoch_freq 20 --init_type xavier --init_gain 0.02
