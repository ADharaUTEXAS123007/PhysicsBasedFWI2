
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/AutoModelFaultPhy/*.txt
no_proxy=localhost python trainValLatent4dVel2.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/Marmousi/ --name Marmousi --model Auto2 --direction AtoB --input_nc 16 --output_nc 1 --display_port 9996 --n_epochs 100000 --n_epochs_decay 0  --batch_size 8 --gpu_ids 0,1,2,3,4,5,6,7 --no_html --display_freq 1 --print_freq 1 --lr 0.001 --verbose --save_epoch_freq 100
