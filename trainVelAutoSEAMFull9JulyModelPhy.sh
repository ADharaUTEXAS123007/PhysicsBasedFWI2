rm -rf ./marmousiSEAM9July
mkdir ./marmousiSEAM9July
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousSEAM9July/*.txt
no_proxy=localhost python trainValLatent4dVel2Elastic.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/SEAM/ --name marmousiSEAM9July --model AutoSEAMMar22 --direction AtoB --input_nc 39 --output_nc 1 --display_port 9998 --n_epochs 4000  --n_epochs_decay 2500 --batch_size 8 --gpu_ids 7 --no_html --display_freq 1 --print_freq 1 --lr 0.001 --verbose --save_epoch_freq 30 --init_type normal --epoch 420 --continue_train
