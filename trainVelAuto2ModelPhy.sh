
#rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/Auto2SaltPhy64/*.txt
no_proxy=localhost python trainValLatent4dVel2.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/Marmousi/ --name AutoOpenFWI --model Auto2 --direction AtoB --input_nc 16 --output_nc 1 --display_port 9996 --n_epochs 400 --n_epochs_decay 200  --batch_size 1 --gpu_ids 3 --no_html --display_freq 1 --print_freq 1 --lr 0.01 --verbose --save_epoch_freq 600
