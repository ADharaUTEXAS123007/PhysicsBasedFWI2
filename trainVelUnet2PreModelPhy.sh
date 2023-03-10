
#rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/Auto2SaltPhy64/*.txt
mkdir ./marmousi2
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiOpenFWI2/*.txt
no_proxy=localhost python trainValLatent4dVel2.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiI/ --name MarmousiOpenFWI2 --model Unet2Pre --direction AtoB --input_nc 1 --output_nc 1 --display_port 8881 --n_epochs 2000 --n_epochs_decay 2000  --batch_size 8 --gpu_ids 3 --no_html --display_freq 1 --print_freq 1 --lr 0.01 --verbose --save_epoch_freq 25
