
#rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/Auto2SaltPhy64/*.txt
mkdir ./vae3
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiOpenFWI3Vae/*.txt
no_proxy=localhost python trainValLatent4dVel2.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiF/ --name MarmousiOpenFWI3Vae --model Vae3 --direction AtoB --input_nc 18 --output_nc 1 --display_port 9990 --n_epochs 2000 --n_epochs_decay 2000  --batch_size 8 --gpu_ids 7 --no_html --display_freq 1 --print_freq 1 --lr 0.01 --verbose --save_epoch_freq 25
