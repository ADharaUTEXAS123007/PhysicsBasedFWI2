
#rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/Auto2SaltPhy64/*.txt
mkdir ./vae2
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiOpenFWI2Vae/*.txt
no_proxy=localhost python trainValLatent4dVel2.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/Marmousi/ --name MarmousiOpenFWI2Vae --model Vae2 --direction AtoB --input_nc 18 --output_nc 1 --display_port 9997 --n_epochs 1000 --n_epochs_decay 1000  --batch_size 8 --gpu_ids 6 --no_html --display_freq 1 --print_freq 1 --lr 0.005 --verbose --save_epoch_freq 25
