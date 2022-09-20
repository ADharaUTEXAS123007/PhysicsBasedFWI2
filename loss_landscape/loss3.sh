#mpirun -np 36  /disk/student/adhara/WORK/DeniseFWI/virginFWI/DENISE-Black-Edition/bin/denise  ./LOSS_CURVE_DATA/seis.inp ./LOSS_CURVE_DATA/seis_fwi.inp
python plot_surface2.py --cuda --model elasticnet --x=-1:1:20 --y=-1:1:20 --model_file /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl22/100_net_G.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --surf_file elastic19Sep100.h5 --plot
#plot_2D.plot_2d_contour('elastic19Sep.h5','train_loss',2.65,20.69,0.4,True)
