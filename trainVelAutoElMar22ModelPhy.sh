rm -rf ./marmousiEl
mkdir ./marmousiEl
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiElMar22/*.txt
no_proxy=localhost python trainValLatent4dVel2Elastic.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiEl/ --name MarmousiElMar22 --model AutoElMar22 --direction AtoB --input_nc 25 --output_nc 1 --display_port 8889 --n_epochs 1500 --n_epochs_decay 500 --batch_size 8 --gpu_ids 4 --no_html --display_freq 1 --print_freq 1 --lr 0.005 --verbose --save_epoch_freq 50
