mpirun -n 1 python plot_surface2.py --mpi --cuda --model simplenet --x=-1:1:51 --y=-1:1:51 --model_file /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/SimpleOpenFWI24/50_net_G.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --surf_file newtest42.h5 --plot