
#rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/Auto2SaltPhy64/*.txt
no_proxy=localhost python trainValLatent4dVel2.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/ --name AutoNOpenFWI --model AutoN --direction AtoB --input_nc 8 --output_nc 1 --display_port 9996 --n_epochs 500 --n_epochs_decay 200  --batch_size 8 --gpu_ids 0,1,2,3,4,5,6,7 --no_html --display_freq 1 --print_freq 1 --lr 0.01 --verbose --save_epoch_freq 25 --init_type kaiming
