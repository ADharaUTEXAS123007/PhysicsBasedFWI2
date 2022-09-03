
mkdir ./marmousi24
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiOpenFWI24/*.txt
no_proxy=localhost python trainValLatent4dVel2.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/Simple/ --name SimpleOpenFWI24 --model Simple24 --direction AtoB --input_nc 30 --output_nc 1 --display_port 9988 --n_epochs 500 --n_epochs_decay 3500  --batch_size 8 --gpu_ids 7 --no_html --display_freq 1 --print_freq 1 --lr 0.005 --verbose
